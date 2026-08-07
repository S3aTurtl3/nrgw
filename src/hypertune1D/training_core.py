from typing import Callable, Optional

import os
import time
from dataclasses import dataclass, field
import jax.random as jr
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import wandb

import hashlib

from scriptt import OverfitTracker, PATIENCE_NUM_EPOCHS, nrg_wrapper_saver, WrapperForNNRG


@dataclass
class LossSpec: # SOURCE: Claude
    """Encapsulates everything that differs between the full and baseline losses/logging."""
    
    variant_name: str
    """"full" or "plain" -- used to build the wandb project name."""

    compute_loss: Callable
    """(model, data, loss_key, step) -> (total_loss, aux_dict). aux_dict may be empty. Also, step will be passed in as a jnp.Array scalar"""

    compute_val_loss: Callable
    """(model, loss_key) -> val_loss (scalar)."""

    file_name_fn: Callable
    """(lr, steps, check_for_overfit_every, desc) -> filename string. Extra variant-specific
    hyperparameters (e.g. KE schedule) should already be bound via functools.partial/closure
    before this LossSpec is constructed."""

    extra_wandb_config: dict = field(default_factory=dict)
    """Additional config dict to merge into wandb.init(config=...), e.g. penalty coefficients."""

    should_break_on_overfit: Callable = lambda step: True
    """(step) -> bool. Full variant only breaks once the KE-penalty schedule has fully decayed
    to 0; baseline variant breaks unconditionally as soon as overfitting is detected.""" 




def run_training_loop(
        model: WrapperForNNRG,
        lattice_size: int,
        dataloader, 
        loss_key: jr.PRNGKey,
        lr: float,
        dataset_test, 
        directory_model_saving,
        loss_spec: LossSpec,
        steps,
        weight_decay=1e-5,
        print_every=100,
        check_for_overfit_every=100,
        desc="",
        save_every=1500 ):
    """Generic training loop shared by both the full and baseline NNRG variants.

    All variant-specific behavior (what loss to compute, what to log, how to name
    the saved model file, and when to stop after overfitting is detected) is
    supplied via `loss_spec`.
    """
    optim = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    wandb.init(project=f"nnrg" + loss_spec.variant_name,
               config={"learning_rate": lr,
                       "steps": steps,
                       "weight_decay": weight_decay,
                       "check_for_overfit_every": check_for_overfit_every,
                       "description": desc,
                       "lattice_size": lattice_size,
                       **loss_spec.extra_wandb_config})
    
    tracker = OverfitTracker(patience=PATIENCE_NUM_EPOCHS*dataloader.array.shape[0]/dataloader.batch_size/check_for_overfit_every, min_delta=0.01)
    fname = loss_spec.file_name_fn(lr, steps, check_for_overfit_every, desc)
    pth = os.path.join(directory_model_saving, fname)

    loss_and_grad = eqx.filter_value_and_grad(loss_spec.compute_loss, has_aux=True)
    validation_loss = eqx.filter_jit(loss_spec.compute_val_loss)

    @eqx.filter_jit
    def make_step(model, opt_state, data, loss_key, step: jax.Array):
        (value, aux), grads = loss_and_grad(model, data, loss_key, step)
        loss_key = jr.split(loss_key, 1)[0]
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return value, aux, model, opt_state,loss_key
    
    step = jnp.array(0, dtype=jnp.int32) 
    best_model = None
    best_validation_loss = float('inf')
    loss_messages = []
    loss_key, key_val = jr.split(loss_key, 2)
    overfitting = False
    while step < steps:
        val_loss = None
        start = time.time()
        data = dataloader(step)
        step += 1
        value, aux, model, opt_state, loss_key = make_step(model, opt_state, data, loss_key, step)
        end = time.time()
        if (step % check_for_overfit_every == 0 or step == steps-1 or step==1):
            val_loss = validation_loss(model, key_val)
            key_val = jr.fold_in(key_val, step)
            tracker_verdict = tracker.update(val_loss)
            if tracker_verdict == "stop":
                overfitting = True
                print(f"Overfitting! val loss: {val_loss}")
            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_model = model
        if step % print_every == 0 or step == steps-1:
            aux_str = ", ".join(f"{k}: {float(v)}" for k, v in aux.items())
            loss_msg = f"Step: {step}, Loss: {value}, {aux_str}, Val loss: {val_loss}, Computation_time: {end- start}"
            print(loss_msg)
            loss_messages.append(loss_msg)
            wandb.log(
                {
                    "total_loss": float(value),
                    **{k: float(v) for k, v in aux.items()},
                    "val_loss": float(val_loss) if val_loss is not None else None,
                    "computation_time_per_step": end - start,
                },
                step=step
            )
        if overfitting or ((step % save_every) == 0 or step == steps - 1 or step == 1):
            if best_model is not None:
                nrg_wrapper_saver(pth, {"depth": len(model.nnrg.submodules)}, best_model)
            if overfitting and loss_spec.should_break_on_overfit(step):
                break
    wandb.finish()
    if best_model is None:
        print("best model was None")
        best_model = model
    return best_model, (opt_state, loss_messages)

    