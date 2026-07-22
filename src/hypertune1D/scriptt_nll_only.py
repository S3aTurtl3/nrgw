
"""
NLL-only variant of scriptt.py.

This script reuses as much as possible from the original training script
(`scriptt.py`) by importing its classes/functions directly, and only
re-implements the pieces that must change because the loss function here
is *pure* negative log-likelihood (NLL): no kinetic-energy (KE) penalty
and no MMD-based regularization on the marginal distributions of the
latent variables.

File naming convention
-----------------------
The original script saves model checkpoints with filenames of the form:

    m<hash>.eqx                     (model weights, from get_model_file_name)
    m<lr><steps>_test.eqx           (optimizer state placeholder)
    <hash>.json                     (Ax/hyperparam-tuning output)
    dataset_cache_<hash>.npz        (dataset cache)

To guarantee this script never overwrites/collides with files produced by
the original script, every filename/hash produced here is prefixed with
"nllonly_" and the hash payload string is also seeded with a distinguishing
tag ("nllonly") before hashing. Since MD5 hashes of different input strings
are (for all practical purposes) never equal, and no original-script file
name begins with "nllonly_", there is no possibility of collision.
"""

import argparse
import hashlib
import os
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import wandb
import numpy as np

from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

# ---------------------------------------------------------------------------
# Reuse everything we can from the original script.
#
# NOTE: this assumes the original script has been saved as `scriptt.py`
# (its filename) in the same directory / on the PYTHONPATH, so it can be
# imported as a module named `scriptt`.
# ---------------------------------------------------------------------------
from scriptt import (
    # model / architecture
    WrapperForNNRG,
    WrapperForNNRGSubModule,
    NNRG,
    NNRGSubModule,
    CNF,
    Func,
    ConcatSquash,
    IdentityCNF,
    normal_log_likelihood,
    COARSE_VAR_NAME,
    LOGP_NAME,
    LATENT_VAR_NAME,
    VECTOR_FIELD_SNAPSHOT_NAME,
    DISENTANGLER_CNF_NAME,
    DECIMATOR_CNF_NAME,
    NRG_HYPERPARAM_NAMES,
    create_model_saver,
    load_model,
    nrg_wrapper_saver,
    # loss-function building blocks (we keep the NLL parts, drop KE/marginal)
    llambda,
    NLLLoss_2,
    NLLLoss,
    # data / dataloading
    DataLoader,
    StreamingDataLoader,
    sample_from_continuous_relaxation_1D,
    create_standardized_dataset,
    get_help_finding_int_time,
    compare_model_vs_validation,
    # misc training utilities
    OverfitTracker,
    InferenceInfo,
    ModelInferenceInfo,
    PATIENCE_NUM_EPOCHS,
    LATTICE_SIZE,
    NNRGIsingConfig,
    NNRGUnitTestConfig,
    compute_number_of_latent_vars_being_regularized,
    make_and_save_visualizations_of_best_models,
    get_discrete_samples_from_model,
    get_discrete_samples,
    sample_from_full_nnrg,
    get_depth_of_full_nnrg,
)

# ---------------------------------------------------------------------------
# File-naming helpers (NLL-only variant). All names are namespaced with the
# "nllonly_" prefix so they can never collide with files written by the
# original script's `get_model_file_identifier` / `get_model_file_name`.
# ---------------------------------------------------------------------------

def get_nll_only_model_file_identifier(
    lr: float,
    steps=10000,
    check_for_overfit_every=100,
    desc="",
):
    payload = (
        f"nllonly_m{lr}{steps}s"
        f"_overfitCheck{check_for_overfit_every}_patience{PATIENCE_NUM_EPOCHS}"
        f"_{desc}test"
    )
    return "nllonly_" + hashlib.md5(payload.encode("utf-8")).hexdigest()


def get_nll_only_model_file_name(
    lr: float,
    steps=10000,
    check_for_overfit_every=100,
    desc="",
):
    return (
        get_nll_only_model_file_identifier(
            lr,
            steps,
            check_for_overfit_every,
            desc,
        )
        + ".eqx"
    )


# ---------------------------------------------------------------------------
# Training loop: identical control flow to the original `train_nnrg`, but
# the loss function computed inside `make_step` is *only* NLL. No KE penalty
# term, no marginal-regularization term, and no ke_schedule/coeff_marginal
# arguments are needed at all.
# ---------------------------------------------------------------------------

def train_nnrg_nll_only(
    model: WrapperForNNRGSubModule,
    dataloader: StreamingDataLoader,
    loss_key,
    lr: float,
    coeff_main_loss_term: float,
    num_time_samples,
    dataset_test,
    num_time_samples_test,
    directory_model_saving,
    steps=10000,
    exact_logp=True,
    weight_decay=1e-5,
    print_every=100,
    check_for_overfit_every=100,
    desc="",
    save_every=1500,
):
    optim = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    wandb.init(
        project="vanillaneural-renormalization-group",
        config={
            "learning_rate": lr,
            "steps": steps,
            "weight_decay": weight_decay,
            "check_for_overfit_every": check_for_overfit_every,
            "description": desc,
            "loss_variant": "nll_only",
            "lattice_size": dataloader.array.shape[1] if hasattr(dataloader, "array") else "N/A",
        },
    )

    tracker = OverfitTracker(
        patience=PATIENCE_NUM_EPOCHS * dataloader.array.shape[0] / dataloader.batch_size / check_for_overfit_every,
        min_delta=0.01,
    )

    fname = get_nll_only_model_file_name(
        lr, steps, check_for_overfit_every, desc
    )
    opt_state_fname = f"nllonly_m{lr}{steps}_test.eqx"
    pth = os.path.join(directory_model_saving, fname)
    pth_opt_state = os.path.join(directory_model_saving, opt_state_fname)

    @eqx.filter_value_and_grad(has_aux=True)
    def loss(model, data, loss_key):
        loss_key, key_shots = jr.split(loss_key, 2)
        key_shots = jr.split(key_shots, data.shape[0])

        all_coarse, logpp, _decimator_snapshots, _disentangler_snapshots = jax.vmap(
            lambda m, example, key: llambda(m, example, num_time_samples, key), in_axes=(None, 0, 0)
        )(model, data, key_shots)

        main_loss = NLLLoss_2(all_coarse, logpp)
        total_loss = main_loss
        return total_loss, (main_loss,)

    @eqx.filter_jit
    def validation_loss(model, loss_key):
        key_shots_val, key_val = jr.split(loss_key, 2)
        key_shots_val = jr.split(key_shots_val, dataset_test.shape[0])
        all_coarse_val, logpp_val = jax.vmap(
            lambda m, example, key: llambda(m, example, num_time_samples_test, key), in_axes=(None, 0, 0)
        )(model, dataset_test, key_shots_val)[:2]
        val_loss = NLLLoss_2(all_coarse_val, logpp_val)
        return val_loss

    @eqx.filter_jit
    def make_step(model: WrapperForNNRGSubModule, opt_state, data, loss_key):
        (value, (main_loss,)), grads = loss(model, data, loss_key)
        loss_key = jr.split(loss_key, 1)[0]
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return value, main_loss, model, opt_state, loss_key

    step = 0
    best_model = None
    best_loss = float("inf")
    loss_msgs = []
    loss_key, key_val = jr.split(loss_key, 2)
    while step < steps:
        val_loss = None
        start = time.time()

        data = dataloader(step)
        step = step + 1
        value, main_loss, model, opt_state, loss_key = make_step(model, opt_state, data, loss_key)

        end = time.time()
        if (step % check_for_overfit_every == 0) or steps == steps - 1:
            val_loss = validation_loss(model, key_val)
            key_val = jr.fold_in(key_val, step)
            tracker_verdict = tracker.update(val_loss)
            if tracker_verdict == "stop":
                print(f"Overfitting! val loss: {val_loss}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model
        if (step % print_every) == 0 or step == steps - 1:
            loss_msg = (
                f"Step: {step}, Loss (NLL only): {value}, "
                f"just NLL: {main_loss}, Val loss: {val_loss}, "
                f"Computation time: {end - start}"
            )
            print(loss_msg)
            loss_msgs.append(loss_msg)
            computation_time = end - start
            wandb.log(
                {
                    "total_loss": float(value),
                    "val_loss": float(val_loss) if val_loss is not None else None,
                    "computation_time_per_step": computation_time,
                },
                step=step,
            )
        if (step % save_every) == 0 or step == steps - 1 or step == steps or step == 1:
            nrg_wrapper_saver(pth, {"depth": len(model.nnrg.submodules)}, best_model if best_model is not None else model)
        if tracker.is_overfitting():
            break

    wandb.finish()
    if best_model is None:
        print("Best_model was None")
        best_model = model
    return best_model, (opt_state, loss_msgs)


# ---------------------------------------------------------------------------
# main(): mirrors the original script's `main`, but strips out every
# argument/flag related to the KE penalty and the marginal-regularization
# term, and writes its Ax-tuning output / dataset cache to filenames that
# are namespaced with "nllonly_" so they never collide with the original.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train NNRG model (NLL-only loss).")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--lr_min", type=float, default=0.001, help="min learning rate")
    parser.add_argument("--coeff_main_loss_term_min", type=float, default=1.0, help="min Coefficient for main loss term")
    parser.add_argument("--seed", type=int, default=5678, help="Random seed")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lattice_size", required=True, type=int)
    parser.add_argument("--num_train_samples", required=True, type=int)
    parser.add_argument("--num_test_samples", required=True, type=int)
    parser.add_argument("--num_val_samples", required=True, type=int)
    parser.add_argument("--num_trials", required=True, type=int)
    parser.add_argument("--temp", required=True, type=float)
    parser.add_argument("--out", required=True, help="should be a directory with longterm storage")
    parser.add_argument("--dir_model_weights", required=True, help="e.g. local scratch if model weights are not important to retain")
    parser.add_argument("--check_overfit_every", type=int, default=100)

    # Hyperparameter search bounds -- only the main-loss-term coefficient and
    # learning rate remain relevant, since there is no KE penalty or
    # marginal-regularization term to tune in this variant.
    parser.add_argument("--main_coeff_min", type=float, default=0.5, help="Lower bound for main loss term coefficient search range")
    parser.add_argument("--main_coeff_max", type=float, default=2.0, help="Upper bound for main loss term coefficient search range")

    args = parser.parse_args()

    LATTICE_SIZE_ISING = args.lattice_size
    key = jr.PRNGKey(5678)
    model_key, loader_key, loss_key, test_key, evaluation_key, key_vis = jr.split(key, 6)

    OUTPUT_DIR = args.out
    TEMP_DIR = args.dir_model_weights
    # "nllonly_" prefix guarantees this never collides with the original
    # script's OUTPUT_FILE_NAME (which has no such prefix).
    OUTPUT_FILE_NAME = "nllonly_" + hashlib.md5(f"nllonly_tuning{vars(args)}".encode("utf-8")).hexdigest() + ".json"
    temp_tag = f"nllonly_{args.lattice_size}T{args.temp:g}".replace(".", "p")
    model_saving_dir = os.path.join(TEMP_DIR, "models", temp_tag)
    os.makedirs(model_saving_dir, exist_ok=True)
    OUTPUT_FILE_SUBDIR = os.path.join(OUTPUT_DIR, temp_tag)
    os.makedirs(OUTPUT_FILE_SUBDIR, exist_ok=True)
    OUTPUT_FILE_PTH = os.path.join(OUTPUT_FILE_SUBDIR, OUTPUT_FILE_NAME)
    PLACEHOLDER_ISING_MEAN = jnp.zeros(LATTICE_SIZE_ISING)
    PLACEHOLDER_ISING_STD = jnp.ones(LATTICE_SIZE_ISING)

    INTEGRATED_TIME = None if args.temp == 0 else max(
        get_help_finding_int_time(test_key, args.temp, LATTICE_SIZE_ISING, 1.0, 1.0, n=100), LATTICE_SIZE_ISING
    )
    COEFF_FOR_BURN_IN = None if args.temp == 0 else 2
    BURN_IN = None if args.temp == 0 else COEFF_FOR_BURN_IN * INTEGRATED_TIME

    NUM_TRAIN_SAMPLES = args.num_train_samples
    NUM_SAMPLES_TEST = args.num_test_samples
    NUM_SAMPLES_VALIDATION = args.num_val_samples
    NUM_CHAINS = 100
    assert (
        NUM_TRAIN_SAMPLES % NUM_CHAINS == 0
        and NUM_SAMPLES_VALIDATION % NUM_CHAINS == 0
        and NUM_SAMPLES_TEST % NUM_CHAINS == 0
    )

    # Dataset cache filename is also namespaced with "nllonly_" so it will
    # never collide with (or accidentally be reused from) the original
    # script's "dataset_cache_<hash>.npz" files -- even though, in practice,
    # reusing the identical dataset would be harmless, this keeps every
    # artifact produced by this script clearly separated from the original.
    DATA_CACHE_FILE_NAME = "nllonly_dataset_cache_" + hashlib.md5(
        f"nllonly_data{LATTICE_SIZE_ISING}_{args.temp}_{NUM_TRAIN_SAMPLES}_{NUM_SAMPLES_TEST}_{NUM_SAMPLES_VALIDATION}_{args.seed}".encode(
            "utf-8"
        )
    ).hexdigest() + ".npz"
    DATA_CACHE_PATH = os.path.join(OUTPUT_DIR, DATA_CACHE_FILE_NAME)

    if os.path.exists(DATA_CACHE_PATH):
        cached = np.load(DATA_CACHE_PATH)
        full_dataset = jnp.array(cached["full_dataset"])
        dataset_mean = jnp.array(cached["dataset_mean"])
        dataset_std = jnp.array(cached["dataset_std"])
        test_dataset = jnp.array(cached["test_dataset"])
        validation_dataset = jnp.array(cached["validation_dataset"])
        loader_key = jr.fold_in(loader_key, 0)
    else:
        dataset_key, test_key_new, loader_key = jr.split(loader_key, 3)

        all_data = sample_from_continuous_relaxation_1D(
            dataset_key,
            NUM_TRAIN_SAMPLES + NUM_SAMPLES_TEST + NUM_SAMPLES_VALIDATION,
            LATTICE_SIZE_ISING,
            args.temp,
            INTEGRATED_TIME,
            BURN_IN,
            NUM_CHAINS,
        )

        all_data, dataset_mean, dataset_std = create_standardized_dataset(all_data)
        full_dataset = all_data[:NUM_TRAIN_SAMPLES]
        test_dataset = all_data[NUM_TRAIN_SAMPLES : NUM_TRAIN_SAMPLES + NUM_SAMPLES_TEST]
        validation_dataset = all_data[
            NUM_TRAIN_SAMPLES + NUM_SAMPLES_TEST : NUM_TRAIN_SAMPLES + NUM_SAMPLES_TEST + NUM_SAMPLES_VALIDATION
        ]

        np.savez(
            DATA_CACHE_PATH,
            full_dataset=np.asarray(full_dataset),
            dataset_mean=np.asarray(dataset_mean),
            dataset_std=np.asarray(dataset_std),
            test_dataset=np.asarray(test_dataset),
            validation_dataset=np.asarray(validation_dataset),
        )

    dataloader = DataLoader(full_dataset, NNRGIsingConfig.BATCH_SIZE, loader_key)

    client = Client()

    LR_PARAM_NAME = "learning rate"
    NLL_METRIC_NAME = "NLL"
    NUM_SAMPLES_BASIC_EVAL = 1000

    def get_description_of_job():
        return "nllonly_" + str(args) + f"lsize{LATTICE_SIZE_ISING}"

    client.configure_parameters(
        parameters=[
            RangeParameterConfig(name=LR_PARAM_NAME, parameter_type="float", bounds=(args.lr_min, args.lr_min), scaling="log"),
        ]
    )
    client.configure_optimization(objective=f"-{NLL_METRIC_NAME}")

    for _ in range(args.num_trials):
        trials = client.get_next_trials(max_trials=1)
        for trial_index, parameters in trials.items():
            lr = parameters[LR_PARAM_NAME]
            coeff_main_loss_term = 1

            model_key_trial = jr.fold_in(model_key, trial_index)
            loss_key_trial = jr.fold_in(loss_key, trial_index)

            model = WrapperForNNRGSubModule(key=model_key_trial)

            best_model, (opt_state, loss_msgs) = train_nnrg_nll_only(
                model,
                dataloader,
                loss_key_trial,
                lr,
                coeff_main_loss_term,
                args.num_time_samples,
                test_dataset,
                args.num_time_samples_evaluation,
                model_saving_dir,
                steps=args.steps,
                check_for_overfit_every=args.check_overfit_every,
                desc=get_description_of_job(),
            )

            eval_key_trial = jr.fold_in(evaluation_key, trial_index)
            key_shots_eval = jr.split(eval_key_trial, validation_dataset.shape[0])
            all_coarse_eval, logpp_eval = jax.vmap(
                lambda m, example, key: llambda(m, example, args.num_time_samples_evaluation, key),
                in_axes=(None, 0, 0),
            )(best_model, validation_dataset, key_shots_eval)[:2]
            nll_eval = float(NLLLoss_2(all_coarse_eval, logpp_eval))

            client.complete_trial(trial_index=trial_index, raw_data={NLL_METRIC_NAME: nll_eval})

    import json
    from collections.abc import Mapping
    from typing import Any

    def make_json_serializable(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return {key: make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_json_serializable(item) for item in obj]
        else:
            return obj

    best_parameters, _ = client.get_best_parameterization() if hasattr(client, "get_best_parameterization") else []

    comparison_dataset = test_dataset[:NUM_SAMPLES_BASIC_EVAL] * dataset_std + dataset_mean
    key_discrete_model, key_discrete_test = jr.split(key_vis)
    name_of_model = get_nll_only_model_file_name(best_parameters[LR_PARAM_NAME], args.steps, args.check_overfit_every, get_description_of_job()) # LEFT OFF
    nrg_model = load_model(os.path.join(model_saving_dir, name_of_model), WrapperForNNRG)
    configs_sampled_from_model = get_discrete_samples_from_model(nrg_model, dataset_mean, dataset_std, key_discrete_model, LATTICE_SIZE_ISING, NUM_SAMPLES_BASIC_EVAL)
    configs_from_test_dataset = get_discrete_samples(comparison_dataset, key_discrete_test)
    fig, stats = compare_model_vs_validation(configs_sampled_from_model, configs_from_test_dataset, n_show=20)
    fname = "OutputVis" + get_nll_only_model_file_identifier(best_parameters[LR_PARAM_NAME], args.steps, args.check_overfit_every, get_description_of_job()) + ".pdf"
    fig.savefig(os.path.join(OUTPUT_FILE_SUBDIR, fname))

    with open(OUTPUT_FILE_PTH, "w") as f:
        json.dump(make_json_serializable({"args": vars(args), "best_parameters": str(best_parameters)}), f)


if __name__ == "__main__":
    main()
