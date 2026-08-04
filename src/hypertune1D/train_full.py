"""
FULL (regularized) NNRG training script.

Differences from train_baseline.py, made explicit via the LossSpec passed
to `training_core.run_training_loop`:

    - Loss = coeff_main_loss_term * NLL
             + coeff_marginal_regularization * marginal_MMD_penalty
             + ke_schedule(step) * kinetic_energy_penalty
      (baseline.py's loss is just NLL, no penalty terms at all)
    - Requires computing vector-field snapshots every step (needed for the
      KE penalty) via `model.inference(...)`, whereas the baseline calls
      the cheaper `model.inference_without_vector_field_snapshots(...)`.
    - Ax hyperparameter search tunes 4 extra dimensions here (KE penalty
      coefficient + its decay schedule length, marginal-regularization
      coefficient, main-loss-term coefficient) that the baseline does not
      search over at all -- baseline only tunes learning rate.
    - Optimization objectives: NLL, MMD, AND the KE penalty on test data
      (3 objectives) vs. baseline's NLL + MMD (2 objectives).
"""

import argparse
import hashlib
import json
import os
import sys
import warnings

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np

from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

from losses import just_nll
from utils import get_unique_identifier
from scriptt import (load_model,
    WrapperForNNRG,
    COARSE_VAR_NAME,
    LOGP_NAME,
    VECTOR_FIELD_SNAPSHOT_NAME,
    DISENTANGLER_CNF_NAME,
    DECIMATOR_CNF_NAME,
    create_standardized_dataset,
    sample_from_continuous_relaxation_1D,
    DataLoader,
     NLLLoss_2,
    llambda,
    kinetic_energy_penalty,
    _regularization_on_marginals,
    make_json_serializable,
    penalties_on_test_data,
    ModelInferenceInfo,
    evaluate_sample_quality_nnrg,
    get_discrete_samples_from_model,
    get_discrete_samples,
    compare_model_vs_validation,
    PATIENCE_NUM_EPOCHS,
    KESchedule,
    NNRGIsingConfig,
    get_model_file_identifier,
    get_model_file_name,
    compute_number_of_latent_vars_being_regularized,
    get_help_finding_int_time, 
)

from training_core import LossSpec, run_training_loop




def train_nnrg(model: WrapperForNNRG,
               dataloader,
               loss_key,
               lr: float,
               coeff_marginal_regularization: float,
               coeff_main_loss_term: float,
               num_time_samples,
               dataset_test,
               num_time_samples_test,
               ke_schedule: KESchedule,
               directory_model_saving,
               steps=10000,
               exact_logp=True,
               weight_decay=1e-5,
               print_every=100,
               check_for_overfit_every=100,
               desc="",
                save_every=1500):
    def compute_loss(model, data, loss_key, step):
        key_reg, key_ke, key_shots = jr.split(loss_key, 3)
        key_shots = jr.split(key_shots, data.shape[0])

        all_coarse, logpp, per_submodule_decimator_vector_field_snapshots, per_submodule_disentangler_vf_snapshots, thing = jax.vmap(lambda m, example, key: jax.checkpoint(llambda, static_argnums=(2,))(m, example, num_time_samples, key), in_axes=(None, 0, 0))(model, data, key_shots)

        keys_ke = jr.split(key_ke, all_coarse.shape[0])

        penalty = jax.vmap(lambda deci_shots, disen_shots, key: jax.checkpoint(kinetic_energy_penalty)(model, deci_shots, disen_shots, key))(per_submodule_decimator_vector_field_snapshots,
                                                                                                                                                                per_submodule_disentangler_vf_snapshots, keys_ke)
        penalty = jnp.mean(penalty)

        marginal_regularization_penalty = _regularization_on_marginals(thing)
        main_loss = NLLLoss_2(all_coarse, logpp)
        total_loss = coeff_main_loss_term*main_loss + coeff_marginal_regularization*marginal_regularization_penalty + ke_schedule.get_next(step)*penalty # optimization improvement: lamdba within jit
        return total_loss, (penalty, marginal_regularization_penalty, main_loss)
    
    loss_spec = LossSpec(
        variant_name="full",
        compute_loss=compute_loss,
        compute_val_loss=lambda model, loss_key: just_nll(model, dataset_test),
        file_name_fn=lambda lr_, steps_, cofe_, desc_: get_model_file_name(lr_, ke_schedule, coeff_marginal_regularization, coeff_main_loss_term, num_time_samples, num_time_samples_test, steps_, cofe_, desc_),
        extra_wandb_config={
            "ke_penalty_coeff_initial": ke_schedule.coeff,
            "ke_penalty_num_steps_till_0": ke_schedule.num_steps_till_0,
            "coeff_marg_reg": coeff_marginal_regularization,
            "coeff_main_loss_term": coeff_main_loss_term,
            "num_time_samples": num_time_samples,
            "num_time_samples_test": num_time_samples_test
        },
        should_break_on_overfit=lambda step: ke_schedule.get_next(step) == 0
    )
    return run_training_loop(model, dataloader.array.shape[1], dataloader, loss_key, lr, dataset_test, directory_model_saving, loss_spec, steps=steps, weight_decay=weight_decay, print_every=print_every,
                             check_for_overfit_every=check_for_overfit_every, desc=desc, save_every=save_every)


def main():
    parser = argparse.ArgumentParser(description='Train NNRG model.')
    parser.add_argument('--batch_size', type=int, required=True,help='Batch size')
    parser.add_argument('--lr_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--ke_penalty_coeff_min', type=float, default=1.0, help='min possible value for KE penalty coefficient')
    parser.add_argument('--coeff_marginal_regularization_min', type=float, default=6.0, help='min Coefficient for marginal regularization')
    parser.add_argument('--coeff_main_loss_term_min', type=float, default=1.0, help='min Coefficient for main loss term')
    parser.add_argument('--num_time_samples', type=int, default=40, help='Number of time samples')
    parser.add_argument('--num_time_samples_evaluation', type=int, default=40, help='Number of time samples')
    parser.add_argument('--seed', type=int, default=5678, help='Random seed')
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--lattice_size', required=True,type=int)
    parser.add_argument('--num_train_samples', required=True,type=int)
    parser.add_argument('--num_test_samples',required=True, type=int)
    parser.add_argument('--num_val_samples', required=True,type=int)
    parser.add_argument('--num_trials', required=True,type=int)
    parser.add_argument('--temp', required=True,type=float) # EFF: add burn in as a parameter else tune
    parser.add_argument('--out', required=True,help='should be a directory with longterm storage (so not local scratch)')
    parser.add_argument('--dir_model_weights', required=True, help='e.g. local scratch if it is not important to retain model weights')
    parser.add_argument('--check_overfit_every', type=int, default=100)

    # Hyperparameter search bounds (excludes learning rate, which remains a fixed choice set)
    parser.add_argument('--penalty_coeff_min', type=float, default=1e-3, help='Lower bound for KE penalty coefficient search range')
    parser.add_argument('--penalty_coeff_max', type=float, default=2.0, help='Upper bound for KE penalty coefficient search range')
    parser.add_argument('--margin_min', type=int, default=6, help='Lower bound for marginal regularization coefficient search range')
    parser.add_argument('--margin_max', type=int, default=10, help='Upper bound for marginal regularization coefficient search range')
    parser.add_argument('--main_coeff_min', type=float, default=0.5, help='Lower bound for main loss term coefficient search range')
    parser.add_argument('--main_coeff_max', type=float, default=2.0, help='Upper bound for main loss term coefficient search range')
    parser.add_argument('--steps_til_0_min', type=int, default=None, help='Lower bound (in steps) for KE-penalty decay schedule length; defaults to 3 epochs worth of steps if not provided')
    parser.add_argument('--steps_til_0_max', type=int, default=None, help='Upper bound (in steps) for KE-penalty decay schedule length; defaults to PATIENCE_NUM_EPOCHS/3 epochs worth of steps if not provided')
    parser.add_argument('--n', type=int, default=2000)

    args = parser.parse_args()

    LATTICE_SIZE_ISING = args.lattice_size
    # Setup keys
    seed = 5678
    key = jr.PRNGKey(seed)
    model_key, loader_key, loss_key, test_key, evaluation_key, key_validation = jr.split(key, 6)

    OUTPUT_DIR = args.out
    TEMP_DIR = args.dir_model_weights
    run_name = get_unique_identifier(vars(args))
    OUTPUT_FILE_NAME = "tuning" + run_name+ ".json"
    temp_tag = f"{args.lattice_size}T{args.temp:g}".replace(".", "p")
    model_saving_dir = os.path.join(TEMP_DIR, "models", temp_tag)
    os.makedirs(model_saving_dir, exist_ok=True)
    OUTPUT_FILE_SUBDIR = os.path.join(OUTPUT_DIR, temp_tag)
    os.makedirs(OUTPUT_FILE_SUBDIR, exist_ok=True)
    OUTPUT_FILE_PTH = os.path.join(OUTPUT_FILE_SUBDIR, OUTPUT_FILE_NAME)
    PLACEHOLDER_ISING_MEAN = jnp.zeros(LATTICE_SIZE_ISING)
    PLACEHOLDER_ISING_STD = jnp.ones(LATTICE_SIZE_ISING)
    # Constants calculation
    OLD_BATCH_SIZE = 500


    INTEGRATED_TIME = None if args.temp == 0 else max(get_help_finding_int_time(test_key, args.temp, LATTICE_SIZE_ISING, 1.0, 1.0, n=args.n), LATTICE_SIZE_ISING)
    COEFF_FOR_BURN_IN= None if args.temp == 0 else 2
    BURN_IN = None if args.temp == 0 else COEFF_FOR_BURN_IN*INTEGRATED_TIME

    #TAINTED================
    # Generate a dataset of size 19000
    NUM_TRAIN_SAMPLES = args.num_train_samples
    NUM_SAMPLES_TEST = args.num_test_samples
    NUM_SAMPLES_VALIDATION = args.num_val_samples
    NUM_CHAINS = 100
    assert NUM_TRAIN_SAMPLES % NUM_CHAINS == 0 and NUM_SAMPLES_VALIDATION % NUM_CHAINS == 0 and NUM_SAMPLES_TEST % NUM_CHAINS == 0

    # Cache generated train/test/validation datasets to a file in the same directory
    # as the script's final output (OUTPUT_DIR), NOT the model-weights directory.
    # If that cache file already exists, load the datasets from it instead of
    # regenerating them.

    DATA_CACHE_FILE_NAME = "dataset_cache_file" + hashlib.md5(
        f"data{LATTICE_SIZE_ISING}_{args.temp}_{NUM_TRAIN_SAMPLES}_{NUM_SAMPLES_TEST}_{NUM_SAMPLES_VALIDATION}_{seed}_n{args.n}".encode('utf-8')
    ).hexdigest() + ".npz"
    DATA_CACHE_PATH = os.path.join(OUTPUT_DIR, DATA_CACHE_FILE_NAME)

    if os.path.exists(DATA_CACHE_PATH):
        cached = np.load(DATA_CACHE_PATH)
        full_dataset = jnp.array(cached["full_dataset"])
        dataset_mean = jnp.array(cached["dataset_mean"])
        dataset_std = jnp.array(cached["dataset_std"])
        test_dataset = jnp.array(cached["test_dataset"])
        validation_dataset = jnp.array(cached["validation_dataset"])
        loader_key = jr.fold_in(loader_key, 0)  # keep RNG stream consistent w/ non-cached path
    else:
        dataset_key, test_key_new, loader_key = jr.split(loader_key, 3)

        all_data = sample_from_continuous_relaxation_1D(dataset_key, NUM_TRAIN_SAMPLES + NUM_SAMPLES_TEST + NUM_SAMPLES_VALIDATION, LATTICE_SIZE_ISING, args.temp, INTEGRATED_TIME, BURN_IN, NUM_CHAINS)

        
        # Standardize the dataset
        all_data, dataset_mean, dataset_std = create_standardized_dataset(all_data)
        full_dataset = all_data[:NUM_TRAIN_SAMPLES]
        test_dataset = all_data[NUM_TRAIN_SAMPLES:NUM_TRAIN_SAMPLES+NUM_SAMPLES_TEST]
        validation_dataset = all_data[NUM_TRAIN_SAMPLES+NUM_SAMPLES_TEST:NUM_TRAIN_SAMPLES+NUM_SAMPLES_TEST+NUM_SAMPLES_VALIDATION]


        np.savez(
            DATA_CACHE_PATH,
            full_dataset=np.asarray(full_dataset),
            dataset_mean=np.asarray(dataset_mean),
            dataset_std=np.asarray(dataset_std),
            test_dataset=np.asarray(test_dataset),
            validation_dataset=np.asarray(validation_dataset),
        )

    # Instantiate the regular DataLoader
    dataloader = DataLoader(full_dataset, NNRGIsingConfig.BATCH_SIZE, loader_key)
    #=========================


    client = Client()

    LR_PARAM_NAME = "learning rate"
    PARAM_NAME_MARGINAL_REGULARIZATION = "margin"
    PARAM_NAME_MAIN_TERM = "mainCoeff"
    PENALTY_COEFF_NAME = "penaltyCoeff"
    PARAM_NAME_STEPS_TIL_0 = "st0"
    NLL_METRIC_NAME = "NLL"
    MMD_METRIC_NAME = "MMD"
    KE_PENALTY_NAME = "KE"

    def get_description_of_job():
        return run_name + f"lsize{LATTICE_SIZE_ISING}"

    def make_and_save_visualizations_of_best_models(frontier, key_frontier, test_dataset):
        
        key_frontier_visualizations = jr.split(key_frontier, len(frontier))
        NUM_SAMPLES_BASIC_EVAL = 500
        comparison_dataset = test_dataset[:NUM_SAMPLES_BASIC_EVAL] * dataset_std + dataset_mean
        for i, (parameters, metrics, trial_index, arm_name) in enumerate(frontier):
            # visualize model samples compared to test dataset
            key_current_parameterization = key_frontier_visualizations[i]
            key_discrete_model, key_discrete_test = jr.split(key_current_parameterization)
            name_of_model = get_model_file_name(
                        lr= parameters[LR_PARAM_NAME],
                        ke_schedule=KESchedule(parameters[PENALTY_COEFF_NAME], parameters[PARAM_NAME_STEPS_TIL_0]),
                        coeff_marginal_regularization=parameters[PARAM_NAME_MARGINAL_REGULARIZATION],
                        coeff_main_loss_term=parameters[PARAM_NAME_MAIN_TERM],
                        steps=args.steps,
                        desc=get_description_of_job(),
                        num_time_samples = args.num_time_samples,
                        num_time_samples_test=args.num_time_samples_evaluation,
                        )
            pth=os.path.join(model_saving_dir, name_of_model)
            if os.path.getsize(pth) <= 0:
                warnings.warn(f"{pth} is empty")
            nrg_model = load_model(pth, WrapperForNNRG)
            configs_sampled_from_model = get_discrete_samples_from_model(nrg_model, dataset_mean, dataset_std, key_discrete_model, LATTICE_SIZE_ISING, NUM_SAMPLES_BASIC_EVAL)
            configs_from_test_dataset = get_discrete_samples(comparison_dataset, key_discrete_test)
            fig, stats = compare_model_vs_validation(configs_sampled_from_model, configs_from_test_dataset, n_show=10)
            fname = "OutputVis" + get_model_file_identifier(lr= parameters[LR_PARAM_NAME],
                        ke_schedule=KESchedule(parameters[PENALTY_COEFF_NAME], parameters[PARAM_NAME_STEPS_TIL_0]),
                        coeff_marginal_regularization=parameters[PARAM_NAME_MARGINAL_REGULARIZATION],
                        coeff_main_loss_term=parameters[PARAM_NAME_MAIN_TERM],
                        steps=args.steps,
                        desc=get_description_of_job(),
                        num_time_samples = args.num_time_samples,
                        num_time_samples_test=args.num_time_samples_evaluation) + ".pdf"
            fig.savefig(os.path.join(OUTPUT_FILE_SUBDIR, fname))



    # Configure and experiment with the desired parameters
    STEPS_IN_EPOCH = int(jnp.ceil(dataloader.array.shape[0]/dataloader.batch_size))

    # Bounds for the KE-penalty decay schedule length default to the original
    # heuristic (3 epochs to PATIENCE_NUM_EPOCHS/3 epochs) unless overridden via CLI.
    steps_til_0_min = args.steps_til_0_min if args.steps_til_0_min is not None else 3 * STEPS_IN_EPOCH
    steps_til_0_max = args.steps_til_0_max if args.steps_til_0_max is not None else int(jnp.ceil(PATIENCE_NUM_EPOCHS * STEPS_IN_EPOCH / 3))

    client.configure_experiment(parameters=[
        RangeParameterConfig(
            name="penaltyCoeff",
            bounds=(args.penalty_coeff_min, args.penalty_coeff_max),
            parameter_type="float",
            scaling="log"
        ),
        RangeParameterConfig(
            name=PARAM_NAME_MARGINAL_REGULARIZATION,
            bounds=(args.margin_min, args.margin_max),
            parameter_type="int",
        ),
        RangeParameterConfig(
            name="mainCoeff",
            bounds=(args.main_coeff_min, args.main_coeff_max),
            parameter_type="float",
            scaling="log"
        ),
        RangeParameterConfig(
        name=PARAM_NAME_STEPS_TIL_0,
        bounds=(steps_til_0_min, steps_til_0_max), # from 3 epochs to half (defaults; overridable via CLI)
        parameter_type="int",
    ),

        ChoiceParameterConfig(
                name="learning rate",
                parameter_type="float",
                values=[
                    1e-3/4,
                    1e-3/4*2,
                    1e-3/4*2**2,
                    1e-3/4*2**3,

                ],
                is_ordered=True,
            ),
    ])

    client.configure_optimization(
    objective=f"-{NLL_METRIC_NAME}, -{MMD_METRIC_NAME}, -{KE_PENALTY_NAME}",
    outcome_constraints=[f"{NLL_METRIC_NAME} <= 300", f"{MMD_METRIC_NAME} <= {0.04*compute_number_of_latent_vars_being_regularized(args.lattice_size)}", f"{KE_PENALTY_NAME} <= 0.2"],
)
    
    


    for i in range(args.num_trials):
        evaluation_key, key_nll, key_sample_quality, key_ot_penalty = jr.split(evaluation_key, 4)
        trials = client.get_next_trials(max_trials=1)
        per_trial_loss_msgs = []
        loss_key = jr.fold_in(loss_key, i)
        for trial_index, parameters in trials.items():
            loss_key = jr.fold_in(loss_key, trial_index)
            # Model initialization
            nrg_model = WrapperForNNRG(depth=int(jnp.ceil(jnp.log2(LATTICE_SIZE_ISING))), key=model_key)


    
            nrg_model, (_, loss_msgs) = train_nnrg(
                nrg_model,
                dataloader,
                loss_key,
                lr=parameters[LR_PARAM_NAME],
                coeff_marginal_regularization=parameters[PARAM_NAME_MARGINAL_REGULARIZATION],
                coeff_main_loss_term=parameters[PARAM_NAME_MAIN_TERM],
                num_time_samples=args.num_time_samples,
                dataset_test = validation_dataset,
                num_time_samples_test= args.num_time_samples_evaluation,
                desc=get_description_of_job(),
                steps=args.steps,
                ke_schedule=KESchedule(parameters[PENALTY_COEFF_NAME], parameters[PARAM_NAME_STEPS_TIL_0]),
                directory_model_saving=model_saving_dir,
                check_for_overfit_every=args.check_overfit_every
            )
            per_trial_loss_msgs.append(loss_msgs)
        
            inference_info = ModelInferenceInfo(nrg_model, PLACEHOLDER_ISING_MEAN, PLACEHOLDER_ISING_STD)
            sample_quality = evaluate_sample_quality_nnrg(inference_info, test_dataset, key_sample_quality, LATTICE_SIZE_ISING)
            penalties = penalties_on_test_data(nrg_model, test_dataset, key_ot_penalty, args.num_time_samples_evaluation)
            raw_data = {NLL_METRIC_NAME: float(penalties["nll"]), MMD_METRIC_NAME: float(sample_quality), KE_PENALTY_NAME: float(penalties["ke"])}
            metric_path = os.path.join(OUTPUT_FILE_SUBDIR, get_model_file_identifier(lr= parameters[LR_PARAM_NAME],
                        ke_schedule=KESchedule(parameters[PENALTY_COEFF_NAME], parameters[PARAM_NAME_STEPS_TIL_0]),
                        coeff_marginal_regularization=parameters[PARAM_NAME_MARGINAL_REGULARIZATION],
                        coeff_main_loss_term=parameters[PARAM_NAME_MAIN_TERM],
                        steps=args.steps,
                        desc=get_description_of_job(),
                        num_time_samples = args.num_time_samples,
                        num_time_samples_test=args.num_time_samples_evaluation,) + "__" + "metrics.json")
            with open(metric_path, "w") as file:
                json.dump({"pars": make_json_serializable(parameters), "metrics": raw_data}, file)
            client.complete_trial(trial_index=trial_index, raw_data=raw_data)

    frontier = client.get_pareto_frontier()
    make_and_save_visualizations_of_best_models(frontier, evaluation_key, test_dataset)
    with open(OUTPUT_FILE_PTH, "w") as file:
      json.dump({"frontier": make_json_serializable(frontier), "loss_msgs": per_trial_loss_msgs}, file)



if __name__ == '__main__':
    # To run in Colab without crashing on sys.argv:
    
    if 'ipykernel' in sys.modules:
        # rendering the following as a sys.arvlist: --batch_size=50 --steps=100 --temp=199 --num_trials=3 --num_train_samples=100 --num_test_samples=100 --num_val_samples=100 --lattice_size=32
        sys.argv = ['', '--batch_size=50', '--steps=100', '--temp=199', '--num_trials=3',  '--num_train_samples=100', '--num_test_samples=100', '--num_val_samples=100', '--lattice_size=32', '--out=/', '--dir_model_weights=/' ]
    main()
