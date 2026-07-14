import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
# Ensure all your classes and functions (WrapperForNNRG, train_nnrg, etc.) are imported or defined above this in the real script.


import sys


from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig


import math
import os
import pathlib
import time

import imageio
from collections.abc import Mapping
import json
from typing import Any, Union


import jax
import jax.lax as lax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import norm

import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox


import scipy.stats as stats


import matplotlib.pyplot as plt  # Used for creating static, interactive, and animated visualizations in Python.
from sklearn.datasets import make_circles, make_moons

import torch
import numpy as np
from ignite.metrics import MaximumMeanDiscrepancy

from typing import Callable, Any, Tuple
from abc import ABC, abstractmethod

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
from numpyro.diagnostics import gelman_rubin, autocorrelation, effective_sample_size


#here = pathlib.Path(os.getcwd())
OUTPUT_DIR = "/n/holyscratch01"
siren_model_dir = os.path.join(OUTPUT_DIR, "/models")


## NN Architecture

import jax
import time
import json
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
from jaxtyping import Array, Float, PyTree

def default_bias_init(key, in_features, out_features: int, dtype=jnp.float32):
    """
    Returns the default Pytorch initialization for the bias of a linear layer

    key: jr.PRNG key
    in_features: int
        num input features to the linear layer that uses this bias
    out_features:
        the size of the linear layer's output
    """
    scale = 1/jnp.sqrt(in_features)
    return jr.uniform(key, (out_features,), dtype, minval=-scale, maxval=scale)


def create_linear_with_custom_initialization(in_size, out_size, key,
                 weight_init=None, bias_init=default_bias_init):
        """Return an eqx.nn.Linear whose weight is given by the function `weight_init` and whose `bias` is given
        by the function `bias_init`

          weight_init:
            accepts as input
        the following parameters (in this order): (key: jr.PRNGKey, in_size: int, out_size: int)
        bias_init:
            accepts as input the following parameters (in this order): (key: jr.PRNGKey, in_size: int, out_size: int)
        """
        linear = eqx.nn.Linear(
            in_size, out_size, key=key)
        where_weight = lambda l: l.weight
        where_bias = lambda l: l.bias
        if weight_init is not None:
            replacement_weights = weight_init(key, in_size, out_size)
            if not replacement_weights.shape == linear.weight.shape:
                raise ValueError(f"expecting weight_init to output matrix of shape {linear.weight.shape}")
            linear = eqx.tree_at(where_weight, linear, replacement_weights)
        if bias_init is not None:
            replacement_bias = bias_init(key, in_size, out_size)
            if not replacement_bias.shape == linear.bias.shape:
                raise ValueError(f"Expecting bias_init to output array of shape {linear.bias.shape}")
            linear = eqx.tree_at(where_bias, linear, replacement_bias)
        return linear


KAIMING_MLP_HYPERPARAM_NAMES = ["data_size", "width_size", "depth", "out_size"]
SIREN_HYPERPARAM_NAMES = ["data_size", "width_size", "depth", "out_size", "omega0"]
FUNC_HYPERPARAM_NAMES = ["data_size", "width_size", "depth"]
NRG_HYPERPARAM_NAMES = ["depth"]

def create_model_saver(hyperparam_names):
    def save(filename, hyperparams, model):
        all_keys_exist = all(key in hyperparams for key in hyperparam_names)
        if not all_keys_exist:
            raise ValueError(f"expected `hyperparams` to have the keys {hyperparam_names}")
        with open(filename, "wb") as f:
            hyperparam_str = json.dumps(hyperparams)
            f.write((hyperparam_str + "\n").encode())
            eqx.tree_serialise_leaves(f, model)
    return save

def load_model(filename, model_class):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = model_class(**hyperparams, key=jr.PRNGKey(0))
        return eqx.tree_deserialise_leaves(f, model)

### Baseline

class Func(eqx.Module):
    '''A network = layers of ConcatSquash '''
    layers: list[eqx.nn.Linear]

    def __init__(self, *, data_size, width_size, depth, key, **kwargs):
        super().__init__(**kwargs)
        keys = jr.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(
                ConcatSquash(in_size=data_size, out_size=data_size, key=keys[0])
            )
        else:
            layers.append(
                ConcatSquash(in_size=data_size, out_size=width_size, key=keys[0])
            )
            for i in range(depth - 1):
                layers.append(
                    ConcatSquash(
                        in_size=width_size, out_size=width_size, key=keys[i + 1]
                    )
                )
            layers.append(
                ConcatSquash(in_size=width_size, out_size=data_size, key=keys[-1])
            )
        self.layers = layers

    def __call__(self, t, y, args):
        t = jnp.asarray(t)[None] # [None], when used in index notation, means insert a new axis here (in this case, beginning)
        for layer in self.layers[:-1]:
            y = layer(t, y)
            y = jnn.tanh(y)
        y = self.layers[-1](t, y)
        return y


# Credit: this layer, and some of the default hyperparameters below, are taken from the
# FFJORD repo.
class ConcatSquash(eqx.Module):
    lin1: eqx.nn.Linear
    lin2: eqx.nn.Linear
    lin3: eqx.nn.Linear

    def __init__(self, *, in_size, out_size, key, **kwargs):
        super().__init__(**kwargs)
        key1, key2, key3 = jr.split(key, 3)
        self.lin1 = eqx.nn.Linear(in_size, out_size, key=key1)
        self.lin2 = eqx.nn.Linear(1, out_size, key=key2)
        self.lin3 = eqx.nn.Linear(1, out_size, use_bias=False, key=key3)

    def __call__(self, t, y):
        return self.lin1(y) * jnn.sigmoid(self.lin2(t)) + self.lin3(t)

## Misc


# %%
"""
Overfitting detection via patience-based early stopping for JAX training loops.

Usage:
    tracker = OverfitTracker(patience=10, min_delta=1e-4)

    for epoch in range(num_epochs):
        val_loss = eval_step(params, val_batch)

        status = tracker.update(val_loss)
        print(tracker.summary())

        if status == "stop":
            print("Early stopping triggered.")
            break
"""

import math
from dataclasses import dataclass
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class EpochRecord:
    epoch: int
    val_loss: float
    is_best: bool


# ---------------------------------------------------------------------------
# Core tracker
# ---------------------------------------------------------------------------

"""
Overfitting detection via patience-based early stopping for JAX training loops.

Usage:
    tracker = OverfitTracker(patience=10, min_delta=1e-4)

    for epoch in range(num_epochs):
        val_loss = eval_step(params, val_batch)

        status = tracker.update(val_loss)
        print(tracker.summary())

        if status == "stop":
            print("Early stopping triggered.")
            break
"""

import math
from dataclasses import dataclass
from typing import Literal, Optional


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class EpochRecord:
    epoch: int
    val_loss: float
    is_best: bool


# ---------------------------------------------------------------------------
# Core tracker
# ---------------------------------------------------------------------------

class OverfitTracker:
    """
    Patience-based early-stopping tracker for JAX training loops.

    Each call to `update` records the validation loss for that epoch and
    returns a status string indicating whether to keep training or stop.

    The concept of patience comes from Prechelt (1998), who introduced
    early stopping as a regularisation technique: training is halted when
    the validation error has not improved for a given number of consecutive
    epochs ("patience"), preventing the model from continuing to overfit.

    Parameters
    ----------
    patience : int
        Number of consecutive epochs without a val-loss improvement
        (by at least `min_delta`) before recommending an early stop.
        Prechelt's original formulation used a strip-based criterion;
        the patience variant used here follows the now-standard practice
        described in Goodfellow et al. (2016), Deep Learning, Ch. 7.
    min_delta : float
        Minimum decrease in val loss to qualify as an improvement.
        Prevents near-flat plateaus from resetting the counter
        indefinitely.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
    ) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")

        self.patience = patience
        self.min_delta = min_delta

        self._history: list[EpochRecord] = []
        self._best_val: float = math.inf
        self._epochs_no_improve: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, val_loss: float) -> Literal["ok", "stop"]:
        """
        Record the validation loss for the current epoch.

        Returns
        -------
        "ok"   – patience not yet exhausted; continue training.
        "stop" – no improvement for `patience` epochs; recommend stopping.

        Parameters
        ----------
        val_loss : float
            Mean validation loss for this epoch. Accepts JAX arrays
            transparently via implicit float() conversion.
        """
        val_loss = float(val_loss)
        epoch = len(self._history)

        improved = val_loss < self._best_val - self.min_delta
        if improved:
            self._best_val = val_loss
            self._epochs_no_improve = 0
        else:
            self._epochs_no_improve += 1

        self._history.append(
            EpochRecord(epoch=epoch, val_loss=val_loss, is_best=improved)
        )

        if self._epochs_no_improve >= self.patience:
            return "stop"
        return "ok"

    def is_overfitting(self) -> bool:
        """True once patience has been exhausted."""
        return self._epochs_no_improve >= self.patience

    def summary(self) -> str:
        """Human-readable one-liner for the current epoch."""
        if not self._history:
            return "No data recorded yet."
        r = self._history[-1]
        return (
            f"Epoch {r.epoch:>4d} | "
            f"val={r.val_loss:.6f}  "
            f"best={self._best_val:.6f}  "
            f"no-improve={self._epochs_no_improve}/{self.patience}"
        )

    def report(self) -> dict:
        """
        Structured diagnostic dictionary for the current epoch.

        Suitable for logging to experiment trackers (W&B, MLflow, etc.)
        or for programmatic use in hyperparameter search loops.
        """
        if not self._history:
            return {}
        r = self._history[-1]
        return {
            "epoch":              r.epoch,
            "val_loss":           r.val_loss,
            "best_val_loss":      self._best_val,
            "epochs_no_improve":  self._epochs_no_improve,
            "patience":           self.patience,
            "recommend_stop":     self.is_overfitting(),
        }

    def reset(self) -> None:
        """Clear all history and reset internal state."""
        self._history.clear()
        self._best_val = math.inf
        self._epochs_no_improve = 0

    @property
    def best_val_loss(self) -> float:
        """Best validation loss seen so far."""
        return self._best_val

    @property
    def history(self) -> list[EpochRecord]:
        """Read-only list of all recorded epoch records."""
        return list(self._history)


# ---------------------------------------------------------------------------
# Convenience function for post-hoc analysis
# ---------------------------------------------------------------------------

def check_overfit(
    val_losses: list[float],
    patience: int = 10,
    min_delta: float = 1e-4,
) -> dict:
    """
    Analyse a completed (or partial) training run from a val-loss list.

    Parameters
    ----------
    val_losses : list[float]
        Per-epoch validation losses.
    patience, min_delta :
        Forwarded to :class:`OverfitTracker`.

    Returns
    -------
    dict with keys: ``epochs``, ``final_report``, ``first_stop_epoch``,
    ``best_val_loss``, ``records``.
    """
    tracker = OverfitTracker(patience=patience, min_delta=min_delta)
    first_stop_epoch: Optional[int] = None

    for i, vl in enumerate(val_losses):
        status = tracker.update(vl)
        if status == "stop" and first_stop_epoch is None:
            first_stop_epoch = i

    return {
        "epochs":           len(val_losses),
        "final_report":     tracker.report(),
        "first_stop_epoch": first_stop_epoch,
        "best_val_loss":    tracker.best_val_loss,
        "records":          tracker.history,
    }



# %%
## Module responsible for the differential equation solve


def approx_logp_wrapper(t, y, args):
    y, _ = y
    *args, eps, func = args
    fn = lambda y: func(t, y, args)
    f, vjp_fn = jax.vjp(fn, y)
    (eps_dfdy,) = vjp_fn(eps)
    logp = jnp.sum(eps_dfdy * eps)
    return f, logp


def exact_logp_wrapper(t, y, args):
    y, _ = y
    *args, _, func = args
    fn = lambda y: func(t, y, args)
    f, vjp_fn = jax.vjp(fn, y)
    (size,) = y.shape  # this implementation only works for 1D input
    eye = jnp.eye(size)
    (dfdy,) = jax.vmap(vjp_fn)(eye)
    logp = jnp.trace(dfdy)
    return f, logp #--- f = func(t, y);;


def normal_log_likelihood(y):
    return -0.5 * (y.size * jnp.log(2 * jnp.pi) + jnp.sum(y**2))



CNFVectorField = Callable[[float, jax.Array, PyTree[Any]], jax.Array]


class CNF(eqx.Module):
    func: eqx.Module
    data_size: int
    exact_logp: bool
    t0: float
    t1: float
    dt0: float

    def __init__(
        self,
        *,
        vector_field_parameterization: eqx.Module,
        data_size,
        exact_logp,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.func = vector_field_parameterization
        self.data_size = data_size
        self.exact_logp = exact_logp
        self.t0 = 0.0
        self.t1 = 0.5
        self.dt0 = 0.05

    # Runs backward-in-time to train the CNF.
    def train(self, y, *, key):
        if self.exact_logp:
            term = diffrax.ODETerm(exact_logp_wrapper)
        else:
            term = diffrax.ODETerm(approx_logp_wrapper)
        solver = diffrax.Tsit5()
        eps = jr.normal(key, y.shape)
        delta_log_likelihood = 0.0
        y = (y, delta_log_likelihood)
        sol = diffrax.diffeqsolve(
            term, solver, self.t1, self.t0, -self.dt0, y, (eps, self.func) # y is passed in as initial condition
        )
        (y,), (delta_log_likelihood,) = sol.ys
        return delta_log_likelihood + normal_log_likelihood(y) #--- normal_log_likelihood is the prior distribution (22-23 min)

    def train_and_compute_latent_variables_and_delta(self, y, *, key):
        if self.exact_logp:
            term = diffrax.ODETerm(exact_logp_wrapper)
        else:
            term = diffrax.ODETerm(approx_logp_wrapper)
        solver = diffrax.Tsit5()
        eps = jr.normal(key, y.shape)
        delta_log_likelihood = 0.0
        y = (y, delta_log_likelihood)
        sol = diffrax.diffeqsolve(
            term, solver, self.t1, self.t0, -self.dt0, y, (eps, self.func)
        )
        (y,), (delta_log_likelihood,) = sol.ys
        return y, delta_log_likelihood

    # Runs forward-in-time to draw samples from the CNF.
    def sample(self, *, key):
        y = jr.normal(key, (self.data_size,))
        term = diffrax.ODETerm(self.func)
        solver = diffrax.Tsit5()
        sol = diffrax.diffeqsolve(term, solver, self.t0, self.t1, self.dt0, y) #--- how to go forward in time
        (y,) = sol.ys
        return y

    def sample_and_compute_density_helper(self, y: jax.Array, term, eps, is_forward_direction: bool):
        """
        eps:
            only is used when computing the approximation of the value of the pdf
            """
        solver = diffrax.Tsit5()
        func = self.func
        args_for_term = (eps, func)
        initial_values_for_y_and_delta_logp = (y, 0.0)

         # Solve CNF ODE (direction determines t0→t1 or t1→t0)
        def forward_branch(_):
            sol = diffrax.diffeqsolve(term, solver, self.t0, self.t1, self.dt0, initial_values_for_y_and_delta_logp, args_for_term)
            return sol.ys

        def backward_branch(_):
            sol =  diffrax.diffeqsolve(term, solver, self.t1, self.t0, -self.dt0, initial_values_for_y_and_delta_logp, args_for_term)
            return sol.ys

        (y_final,), (delta_log_likelihood,) = lax.cond(
            is_forward_direction, forward_branch, backward_branch, operand=None
            )
        return y_final, delta_log_likelihood

    def sample_and_compute_density_exact(self, y, *, is_forward_direction):
        """Returns the tuple (y_final, delta_log_likelihood) where `y_final` is a data point z(t_1) = `y_final` resulting from solving the initial value problem
        Uses the exact computation for the log probability density

        `z(t_0) = y`, `dz(t)/dt = f(z(t), t; θ)`. If `is_forward_direction` is `True`, then `y` is evolved along the ODE forward in time, else
         backward-in-time. `y_final` has the same shape as `y`.

        `exp(d)` is the probability density of the model distribution. `exp(h)` is the pdf of the base distribution, the distribution which the CNF,
        through the change of variables formula, when run forward-in-time, transforms to the model distribution.


        If `is_forward_direction` is `False`, `delta_log_likelihood` is `d - h` where `exp(d)` is the probability density of sampling the provided
        data point `y` from the model distribution.

        y: jax.ndarray
            if `is_forward_direction` is `False,` this variable is assumed to be sampled from the model distribution the latent variable
        key: jax ArrayLike
            a PRNG key for sampling `eps` (used only if self.exact_logp is `False`)
        is_forward_direction: boolean
            If True, the flow is evaluated forward-in-time (latent space -> data space).
            If False, the flow is evaluated backward-in-time (data space -> latent space).
        """
        term = diffrax.ODETerm(exact_logp_wrapper)
        return self.sample_and_compute_density_helper(y, term, jnp.zeros(y.shape), is_forward_direction)


    def sample_and_compute_density(self, y, *, key, is_forward_direction=True):
        """Returns the tuple (y_final, delta_log_likelihood) where `y_final` is a data point z(t_1) = `y_final` resulting from solving the initial value problem

        `z(t_0) = y`, `dz(t)/dt = f(z(t), t; θ)`. If `is_forward_direction` is `True`, then `y` is evolved along the ODE forward in time, else
         backward-in-time. `y_final` has the same shape as `y`.

        `exp(d)` is the probability density of the model distribution. `exp(h)` is the pdf of the base distribution, the distribution which the CNF,
        through the change of variables formula, when run forward-in-time, transforms to the model distribution.


        If `is_forward_direction` is `False`, `delta_log_likelihood` is `d - h` where `exp(d)` is the probability density of sampling the provided
        data point `y` from the model distribution.

        y: jax.ndarray
            if `is_forward_direction` is `False,` this variable is assumed to be sampled from the model distribution the latent variable
        key: jax ArrayLike
            a PRNG key for sampling `eps` (used only if self.exact_logp is `False`)
        is_forward_direction: boolean
            If True, the flow is evaluated forward-in-time (latent space -> data space).
            If False, the flow is evaluated backward-in-time (data space -> latent space).
        """

        if self.exact_logp:
            term = diffrax.ODETerm(exact_logp_wrapper)
        else:
            term = diffrax.ODETerm(approx_logp_wrapper)

        solver = diffrax.Tsit5()
        eps = jr.normal(key, y.shape) # only is used when computing the approximation of the value of the pdf
        func = self.func
        args_for_term = (eps, func)
        initial_values_for_y_and_delta_logp = (y, 0.0)

         # Solve CNF ODE (direction determines t0→t1 or t1→t0)
        def forward_branch(_):
            sol = diffrax.diffeqsolve(term, solver, self.t0, self.t1, self.dt0, initial_values_for_y_and_delta_logp, args_for_term)
            return sol.ys

        def backward_branch(_):
            sol =  diffrax.diffeqsolve(term, solver, self.t1, self.t0, -self.dt0, initial_values_for_y_and_delta_logp, args_for_term)
            return sol.ys

        (y_final,), (delta_log_likelihood,) = lax.cond(
            is_forward_direction, forward_branch, backward_branch, operand=None
            )
        return y_final, delta_log_likelihood


    def get_vector_field_snapshots(self, y, *, is_forward_direction, num_time_samples, key):
      t_so_far = self.t0

      save_times = jnp.sort(jr.uniform(key, (num_time_samples,), minval=self.t0, maxval=self.t1))
      fn = lambda t, y, args: self.func(t, y[0], args)
      snapshots = []
      save_ts = save_times - t_so_far

      term = diffrax.ODETerm(exact_logp_wrapper)
      solver = diffrax.Tsit5()
      saveat = diffrax.SaveAt(dense=True)

      func = self.func
      eps = jnp.zeros(y.shape)
      args_for_term = (eps, func)
      initial_values_for_y_and_delta_logp = (y, 0.0)

        # Solve CNF ODE (direction determines t0→t1 or t1→t0)
      def forward_branch(_):
          sol = diffrax.diffeqsolve(term, solver, self.t0, self.t1, self.dt0, initial_values_for_y_and_delta_logp, args_for_term, saveat=saveat)
          return sol

      def backward_branch(_):
          sol =  diffrax.diffeqsolve(term, solver, self.t1, self.t0, -self.dt0, initial_values_for_y_and_delta_logp, args_for_term, saveat=saveat)
          return sol

      sol = lax.cond(
          is_forward_direction, forward_branch, backward_branch, operand=None
          )
      state_snapshots = jax.vmap(sol.evaluate)(save_ts)[0]
      vector_field_snapshots = jax.vmap(lambda t, y: self.func(t, y, args_for_term))(save_ts, state_snapshots)
      return vector_field_snapshots #vector_field_snapshots is shape (num_time_samples, self.data_size)

    # To make illustrations, we have a variant sample method we can query to see the
    # evolution of the samples during the forward solve.
    def sample_flow(self, *, key):
        t_so_far = self.t0
        t_end = self.t0 + (self.t1 - self.t0)  #--- for us, t1
        save_times = jnp.linspace(self.t0, t_end, 6) #--- save 6 evenly spaced checkpoints- these are the times at which they occur
        y = jr.normal(key, (self.data_size,)) #--- sampled latent var, which has same shape as target var. Note it's not augmented
        out = []
        save_ts = save_times[t_so_far <= save_times] - t_so_far #--- find how far we are from all unpassed checkpoint times

        term = diffrax.ODETerm(self.func)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=save_ts)
        sol = diffrax.diffeqsolve(
            term, solver, self.t0, self.t1, self.dt0, y, saveat=saveat
        )
        out.append(sol.ys)
        y = sol.ys[-1]
        out = jnp.concatenate(out) #--- shape probs 6 by 2
        assert len(out) == 6  # number of points we saved at
        return out



## Neural Network Renormalization Group

class IdentityCNF(eqx.Module):
    """A Placeholder for an instance of CNF. """

    def sample_and_compute_density(self, data: jax.Array, key: jr.PRNGKey, is_forward_direction: bool):
        """Returns a tuple containing:
        a) `data`
        b) 0 """
        return data, jnp.zeros(data.shape[0])

COARSE_VAR_NAME = "coarse"
DECIMATOR_INPUT_NAME = "decin"
DISENTANGLER_INPUT_NAME = "disin"
LOGP_NAME = "l"
LATENT_VAR_NAME = "la"
VECTOR_FIELD_SNAPSHOT_NAME = "vfsnap"
DISENTANGLER_CNF_NAME = "disen"
DECIMATOR_CNF_NAME = "deci"

ACTUAL_COARSE_VAR = "or"
ACTUAL_LATENT_VAR = "lv"



import jax
import jax.numpy as jnp

# TAINTED


def remove_row(arr: jnp.ndarray, row_idx: int) -> jnp.ndarray:
    """Removes a row from a 2D JAX array at the specified index.
    
    Note: For this to work with jax.jit, the row_idx must be a static 
    value or you must accept that the output shape will depend on a dynamic mask.
    """
    # Create a mask of rows to keep
    mask = jnp.arange(arr.shape[0]) != row_idx
    
    # Use advanced indexing with the boolean mask
    return arr[mask, :]

def insert_row(arr: jnp.ndarray, row_idx: int, vec: jnp.ndarray) -> jnp.ndarray:
    """Inserts a 1D vector into a 2D JAX array at the specified row index."""
    # Ensure the vector is treated as a 2D row so shapes align for stacking
    vec_row = jnp.atleast_2d(vec)
    
    # Split and stack the array with the new row in the middle
    return jnp.vstack([arr[:row_idx], vec_row, arr[row_idx:]])



INPUT_SIZE_OF_FLOW_WITHIN_NEURALRG = 2

BEST_WIDTH_SIZE_FFJORD_CNF = 128
"""If the aim is to train a CNF (with the architecutre as described in this tutorial https://docs.kidger.site/diffrax/examples/continuous_normalising_flow/) to model any 2D distribution, this is a good CNF width-size"""
BEST_DEPTH_FFJRD_CNF = 3
"""If the aim is to train a CNF (with the architecutre as described in this tutorial https://docs.kidger.site/diffrax/examples/continuous_normalising_flow/) to model any 2D distribution, this is a good depth"""


# TAINTED!!!!
class NNRGSubModule(eqx.Module):
    """Implements 2 consecutive layers in a Neural Network Renormalization Group (NNRG) (architecture is as described in
    the paper, except for the choice of architecture for the normalizing flow. This code uses continuous normalizing
    flows rather than RealNVPs to construct each layer)

    Assumes that training data is translationally invariant in space"""

    disentangler: CNF
    decimator: CNF
    t0: float
    t1: float


    def __init__(self, *, key: jr.PRNGKey, **kwargs):
        super().__init__(**kwargs)
        disentangler_key, decimator_key = jr.split(key, 2)
        self.disentangler = self._create_default_CNF(disentangler_key)
        self.decimator = self._create_default_CNF(decimator_key)
        assert self.disentangler.t0 == self.decimator.t0
        assert self.disentangler.t1 == self.decimator.t1
        self.t0 = self.disentangler.t0
        self.t1 = self.disentangler.t1

    def generate(self, key: jr.PRNGKey, z: jax.Array):
        """
            Returns a tuple containing:
                1) the latent variables `x` (a 2D array with 2 columns. This represents a 1D lattice– the values of the lattice sites are obtained by flattening `x`) resulting from evolving the latent variable `z` via the 2 consecutive NNRG layers (the layers are evaluated in the forward-in-time direction)
                2) latent variables outputted by the decimator
                3) the variables that are sampled from the standard normal distribution and
                4) the change in log-likelihood
            z: jax.Array
                an 1D array. This represents a 1D lattice (the values of the lattice sites are obtained by flattening `z`)

        """
        variables_other_half = jr.normal(key, z.shape[0])

        z = jnp.stack([variables_other_half, z], axis=1) # be consistent with the side on which you stack/decimate

        z, local_delta_log_likelihood = jax.vmap(lambda data: self.decimator.sample_and_compute_density_exact(data, is_forward_direction=True))(z)

        delta_log_likelihood = local_delta_log_likelihood.sum()

        intra_latents = z
        #Equivalent to z = jnp.roll(z, shift=-1)
        n_sites = z.shape[0]
        col1_indices = (jnp.arange(n_sites) + 1) % n_sites
        z = jnp.stack([
            z[:, 1],               # Current site's second column
            z[col1_indices, 0]     # Next site's first column
        ], axis=1)

        z, local_delta_log_likelihood2 = jax.vmap(lambda data: self.disentangler.sample_and_compute_density_exact(data, is_forward_direction=True))(z)
        delta_log_likelihood += local_delta_log_likelihood2.sum()
        return z, intra_latents, variables_other_half, delta_log_likelihood


    def inference(self, z: jax.Array):
        """
            Returns a tuple containing:
                1) the latent variables `y` (a 2D array with 2 columns. This represents a 1D lattice– the values of the lattice sites are obtained by flattening `y`) resulting from evolving the latent variable `z` via the 2 consecutive NNRG layers in the backwards-in-time direction
                4) the change in log-likelihood (what you add to the log probability density of `y` to get the log probability density of `z`)

            z: jax.Array
                a 2D array with 2 columns. This represents a 1D lattice (the values of the lattice sites are obtained by flattening `z`)"""

        disentangler_input = z
        z, local_delta_log_likelihood = jax.vmap(lambda data: self.disentangler.sample_and_compute_density_exact(data, is_forward_direction=False))(z)
        # compute the change in log likelihood after processing a latent z with the disentangler layer
        delta_log_likelihood = local_delta_log_likelihood.sum()

        # Equivalent to z = jnp.roll(z, shift=1)
        n_sites = z.shape[0]
        col0_indices = (jnp.arange(n_sites) - 1) % n_sites
        z = jnp.stack([
            z[col0_indices, 1],  # Previous site's second column
            z[:, 0]               # Current site's first column
        ], axis=1)

        decimator_input = z
        z, local_delta_log_likelihood2 = jax.vmap(lambda data: self.decimator.sample_and_compute_density_exact(data, is_forward_direction=False))(z)
        delta_log_likelihood += local_delta_log_likelihood2.sum()

        coarse_variables = z[:, 0]
        latent_variables = z[:, 1]


        return {COARSE_VAR_NAME: coarse_variables,
                LATENT_VAR_NAME: latent_variables, DECIMATOR_INPUT_NAME: decimator_input, LOGP_NAME: delta_log_likelihood, DISENTANGLER_INPUT_NAME: disentangler_input}

    def inference_with_vector_field_snapshots(self, z, num_time_samples, key):
      disentangler_input = z
      key, key_for_making_snapshots_dis, key_shot_deci = jr.split(key, 3)
      key_dis_choice, key_deci_choice = jr.split(key)
      selected_index = jr.choice(key_dis_choice, z.shape[0])
      portion_of_z = z[selected_index]

      z, local_delta_log_likelihood = jax.vmap(lambda data: self.disentangler.sample_and_compute_density_exact(data, is_forward_direction=False))(z)
      vector_field_snapshots_disen = self.disentangler.get_vector_field_snapshots(portion_of_z, is_forward_direction=False, num_time_samples = num_time_samples, key=key_for_making_snapshots_dis)



      delta_log_likelihood = local_delta_log_likelihood.sum()

      # Equivalent to z = jnp.roll(z, shift=1)
      n_sites = z.shape[0]
      col0_indices = (jnp.arange(n_sites) - 1) % n_sites
      z = jnp.stack([
          z[col0_indices, 1],  # Previous site's second column
          z[:, 0]               # Current site's first column
      ], axis=1)

      decimator_input = z

      portion_of_z = z[selected_index]

      z, local_delta_log_likelihood2 = jax.vmap(lambda data: self.decimator.sample_and_compute_density_exact(data, is_forward_direction=False))(z)
      portion_of_z = portion_of_z
      vector_field_snapshots_deci = self.decimator.get_vector_field_snapshots(portion_of_z, is_forward_direction=False, num_time_samples=num_time_samples, key=key_shot_deci)


      delta_log_likelihood += local_delta_log_likelihood2.sum()

      coarse_variables = z[:, 0]
      latent_variables =z[:, 1]



      return {COARSE_VAR_NAME: coarse_variables,
                LATENT_VAR_NAME: latent_variables, DECIMATOR_INPUT_NAME: decimator_input, LOGP_NAME: delta_log_likelihood, DISENTANGLER_INPUT_NAME: disentangler_input,
              VECTOR_FIELD_SNAPSHOT_NAME: {DISENTANGLER_CNF_NAME: vector_field_snapshots_disen, DECIMATOR_CNF_NAME: vector_field_snapshots_deci} }
    def _create_default_CNF(self, key: jr.PRNGKey):
        return CNF(vector_field_parameterization= self._create_default_vector_field(key),
        data_size=2,
        exact_logp=True,
        )

    def _create_default_vector_field(self, key: jr.PRNGKey):
        return Func(data_size=INPUT_SIZE_OF_FLOW_WITHIN_NEURALRG,
                            width_size=BEST_WIDTH_SIZE_FFJORD_CNF,
                            depth=BEST_DEPTH_FFJRD_CNF,
                            key=key)



from functools import reduce

class NNRG(eqx.Module):

    submodules: list[NNRGSubModule]

    def __init__(self, *, num_layers, key: jr.PRNGKey, **kwargs):
        keys = jr.split(key, num_layers)
        self.submodules = [NNRGSubModule(key=k) for k in keys ]
        # IT IS ASSUMED ALL SUBMODULES have same t0 and t1

    def inference(self, x):
        """
        Returns a tuple containing:
            - Final latent variables (2D array with 2 columns)
            - total change in log-likelihood (the quantity that must be added to log p(latents) to get log p(x)

        :param self: Description
        :param key: Description
        :param x: a 2D array with 2 columns representing a 1D lattice
        """
        # 1. Prepare keys for each submodule
        num_layers = len(self.submodules)

        total_delta_log_likelihood = 0.0
        current_x = x

        # We will collect the latent variables produced at each step
        latents_list = []
        per_submodule_disentangler_inputs = []
        per_submodule_decimator_inputs = []


        # 2. Unrolled Python Loop
        # JAX will trace this loop and compile it as a straight-line sequence of calls
        for i in range(num_layers):
            layer = self.submodules[i]

            # layer.inference returns:
            # (coarse_variables, latent_variables, delta_log_likelihood)
            current_x = jnp.reshape(current_x, (-1, 2))
            layer_result = layer.inference(current_x)
            current_x = layer_result[LATENT_VAR_NAME]

            # Accumulate results
            total_delta_log_likelihood += layer_result[LOGP_NAME]
            latents_list.append(layer_result[COARSE_VAR_NAME])
            if i == num_layers-1:
              latents_list.append(layer_result[LATENT_VAR_NAME])


        """
        latents_list[-1] = jnp.stack((latents_list[-1], current_x), axis=1).flatten()
        for i in range(num_layers-2, -1, -1):
            latents_list[i] = jnp.stack((latents_list[i], latents_list[i+1]), axis=1).flatten() # result would be in latents_list[0]
        """


        # 3. Consolidate results
        # current_x is now the "final_z" (the most coarse representation)
        # We stack the collected latents along a new axis
        all_coarse_variables = jnp.concatenate(latents_list)

        return {COARSE_VAR_NAME: all_coarse_variables, LOGP_NAME: total_delta_log_likelihood}

    #tainted
    def inference_with_vector_field_snapshots(self, x, num_time_samples, key):
      num_layers = len(self.submodules)

      total_delta_log_likelihood = 0.0
      current_x = x
      key_per_submodule = jr.split(key, len(self.submodules))

      # We will collect the latent variables produced at each step
      latents_list = []
      per_submodule_disentangler_inputs = []
      per_submodule_decimator_inputs = []
      per_submodule_vector_field_snapshots_disentangler = []
      per_submodule_vector_field_snapshots_decimator = []

      # Unrolled Python Loop
      # JAX will trace this loop and compile it as a straight-line sequence of calls

      for i in range(num_layers):
          layer = self.submodules[i]
          key_sub_module = key_per_submodule[i]

          current_x = jnp.reshape(current_x, (-1, 2))
          layer_result = layer.inference_with_vector_field_snapshots(current_x, num_time_samples, key_sub_module)
          current_x = layer_result[LATENT_VAR_NAME]

          total_delta_log_likelihood += layer_result[LOGP_NAME]
          latents_list.append(layer_result[COARSE_VAR_NAME])
          per_submodule_vector_field_snapshots_disentangler.append(layer_result[VECTOR_FIELD_SNAPSHOT_NAME][DISENTANGLER_CNF_NAME])
          per_submodule_vector_field_snapshots_decimator.append(layer_result[VECTOR_FIELD_SNAPSHOT_NAME][DECIMATOR_CNF_NAME])
          if i == num_layers-1:
              latents_list.append(layer_result[LATENT_VAR_NAME])

      all_coarse_variables = jnp.concatenate(latents_list)
      all_vector_field_snapshots_disentangler = jnp.stack(per_submodule_vector_field_snapshots_disentangler)
      all_vector_field_snapshots_decimator = jnp.stack(per_submodule_vector_field_snapshots_decimator)

      return {COARSE_VAR_NAME: all_coarse_variables, LOGP_NAME: total_delta_log_likelihood, VECTOR_FIELD_SNAPSHOT_NAME: {DECIMATOR_CNF_NAME: all_vector_field_snapshots_decimator, DISENTANGLER_CNF_NAME: all_vector_field_snapshots_disentangler}}





    def generate(self, key, z):
        """ Returns a tuple containing:
        - generated samples (2D array with 2 columns)
        - latent variables that should be, via regularization, encouraged to have a gaussian marginal distribut
        - the log_likelihood of a vector containing all coarse-grain variables that are sampled in the code for this function
        - The total change in log-likelihood
        :param z: A 1D array representing the deepest latent variab """
        keys = jr.split(key, len(self.submodules))
        total_delta_log_likelihood = 0.0
        log_likelihood_partial = 0.0
        all_non_final_latents = []

        current_z = z

        to_iterate_over = zip(reversed(self.submodules), keys)
        for i, (layer, layer_key) in enumerate(to_iterate_over):
            current_z = current_z.flatten()
            current_z, intra_latents, coarse_variables, delta_log_likelihood = layer.generate(layer_key, current_z)
            total_delta_log_likelihood += delta_log_likelihood
            log_likelihood_partial += normal_log_likelihood(coarse_variables)
            all_non_final_latents.append(intra_latents)
            all_non_final_latents.append(current_z)


        all_non_final_latents = jnp.concatenate(all_non_final_latents[:-1])
        return current_z, all_non_final_latents, log_likelihood_partial, total_delta_log_likelihood


        """
        keys = jr.split(key, len(self.num))
        total_delta_log_likelihood = 0.0
        all_non_final_latents = None
        current_z = z

        to_iterate_over = zip(reversed(self.submodules), keys)
        for i, (layer, layer_key) in enumerate(to_iterate_over):
            current_z, intra_latents, delta_log_likelihood = layer.generate(layer_key, current_z)
            total_delta_log_likelihood += delta_log_likelihood
            if all_non_final_latents == None:
                all_non_final_latents = intra_latents
            else:
                if i == len(to_iterate_over) - 1:
                    all_non_final_latents = jnp.concatenate((all_non_final_latents, intra_latents))
                else:
                    all_non_final_latents = jnp.concatenate((all_non_final_latents, intra_latents, current_z))

        return current_z, all_non_final_latents, total_delta_log_likelihood
        """

    def generate_with_layers_subset(self, key, z, num_layers):
        """
        num_layers:
            if equal to the depth of this NeuralRG, the physical variables are returned.
            if a > b, the latents returned by  `generate_with_layers_subset(key, z, a)` will be less coarse than the latent
             varibales returned by  """
        keys = jr.split(key, len(self.num))
        total_delta_log_likelihood = 0.0
        current_z = z

        to_iterate_over = zip(reversed(self.submodules), keys)
        for i in range(num_layers):
            layer, layer_key = to_iterate_over[i]
            current_z, intra_latents, delta_log_likelihood = layer.generate(layer_key, current_z)
            total_delta_log_likelihood += delta_log_likelihood

        return current_z, total_delta_log_likelihood



class WrapperForNNRGSubModule(eqx.Module):
    """Implements 2 consecutive layers in a Neural Network Renormalization Group (NNRG) (architecture is as described in
    the paper, except for the choice of architecture for the normalizing flow and the fact that this architecture is speciallized to handle
     1D lattices. This code uses continuous normalizing
    flows rather than RealNVPs to construct each layer)

    Assumes that training data is translationally invariant in space"""

    nnrg: NNRGSubModule

    def __init__(self, *, key: jr.PRNGKey, **kwargs):
        super().__init__(**kwargs)
        self.nnrg = NNRGSubModule(key=key)

    def inference(self, x:jax.Array):
        """
        Returns a tuple containing:
                1) the latent variables `z` (1D array) resulting from evolving the latent variable `x` via the 2 consecutive NNRG layers in the backwards-in-time direction
                2) the change in log-likelihood (what you add to the log probability density of `z` to get the log probability density of `x`)
        x:
            a 1D array"""
        half_of_the_latent_variables, other_half_of_latent_variables, intra_latents, delta_log_likelihood = self.nnrg.inference( jnp.reshape(x, (-1, 2)))
        latent = jnp.stack([half_of_the_latent_variables, other_half_of_latent_variables], axis=1)
        return latent.flatten(), delta_log_likelihood

    def generate(self, key:jr.PRNGKey, z):
        """
        Returns a tuple containing:
                1) the latent variables `x` (1D array) resulting from evolving the latent variable `z` via the 2 consecutive NNRG layers (the layers are evaluated in the forward-in-time direction)
                3) the change in log-likelihood
        z:
            a 1D array"""
        transformed_latent, intra_latents, delta_log_likelihood = self.nnrg.generate(key, z)
        return transformed_latent.flatten(), intra_latents.flatten(), delta_log_likelihood

class WrapperForNNRG(eqx.Module):

    nnrg: NNRG

    def __init__(self, *, depth, key: jr.PRNGKey, **kwargs):
        super().__init__(**kwargs)
        self.nnrg = NNRG(num_layers=depth, key=key)

    def inference(self, x:jax.Array, num_time_samples, key):
        """
        Returns a tuple containing:
        - the latent variables `z` (1D array) resulting from evolving the latent variable `x` in the backwards-in-time direction
        - the change in log-likelihood (what you add to the log probability density of `z` to get the log probability density of `x`)
        x:
            a 1D array"""
        nnrg_output = self.nnrg.inference_with_vector_field_snapshots(jnp.reshape(x, (-1, 2)), num_time_samples, key)
        return {COARSE_VAR_NAME: nnrg_output[COARSE_VAR_NAME].flatten(), LOGP_NAME: nnrg_output[LOGP_NAME], VECTOR_FIELD_SNAPSHOT_NAME: {DISENTANGLER_CNF_NAME: nnrg_output[VECTOR_FIELD_SNAPSHOT_NAME][DISENTANGLER_CNF_NAME], DECIMATOR_CNF_NAME: nnrg_output[VECTOR_FIELD_SNAPSHOT_NAME][DECIMATOR_CNF_NAME]}}


    def generate(self, key:jr.PRNGKey, z):
        """
        Returns a tuple containing:
        - the latent variables `y` resulting from evolving
        - a 1D array whose components are all intended to have a marginal standard gaussian distribution after training the module
        -
        - the change in log-likelihood (what you add to the log probability density of `y` to get the log probability density of `z`)
        """
        transformed_latent, intermediate_latent, log_likelihood_partial, delta_log_likelihood = self.nnrg.generate(key, z)
        return transformed_latent.flatten(), intermediate_latent.flatten(), log_likelihood_partial, delta_log_likelihood

nrg_wrapper_saver = create_model_saver(NRG_HYPERPARAM_NAMES)




















## Data loader

class DataLoader(eqx.Module): #--- generic data loader where data points come in up to several arrays of corresponding elements
    array: jnp.ndarray
    batch_size: int
    key: jr.PRNGKey
    data_size: int

    def __init__(self, array, batch_size, key):
      self.array = array
      self.batch_size = batch_size
      self.key = key
      self.data_size = array.shape[1]

    def __check_init__(self):
        dataset_size = self.array.shape[0]
        assert self.array.shape[0] == dataset_size

    def __call__(self, step):
        """
        step: int
            used to track epoch and batch position; used to determine batch to generate dynamically
            at a given iteration. answers: 'Given this global training step, which batch should exist, and how should
            the data be shuffled?'"""
        dataset_size = self.array.shape[0]
        num_batches = dataset_size // self.batch_size
        epoch = step // num_batches
        key = jr.fold_in(self.key, epoch)
        perm = jr.permutation(key, jnp.arange(dataset_size))
        start = (step % num_batches) * self.batch_size
        slice_size = self.batch_size
        batch_indices = lax.dynamic_slice_in_dim(perm, start, slice_size)
        return self.array[batch_indices]



class StreamingDataLoader:
    get_batch: Callable[[jr.PRNGKey,int], jax.Array]
    """a function with input arguments (key: jr.PRNGKey, batch_size: int) that returns a jax array"""
    batch_size: int
    key: jr.PRNGKey

    def __init__(self, get_batch: Callable, batch_size: int, key: jr.PRNGKey):
        self.get_batch = get_batch
        self.batch_size = batch_size
        self.key = key

    def __call__(self, step):
        key = jr.fold_in(self.key, step)
        return self.get_batch(key, self.batch_size)




CHECKERBOARD_DATASET_MEAN = jnp.array([0, 0])
CHECKERBOARD_DATASET_STD = jnp.array([4/jnp.sqrt(3)]*2)

def sample_from_checkerboard(key, dataset_size):
    x1_key, x2_first_key, x2_second_key = jr.split(key, 3)
    x1 = jr.uniform(x1_key, (dataset_size,)) * 4 - 2
    x2_ = jr.uniform(x2_first_key, (dataset_size,)) - jr.randint(x2_second_key, (dataset_size,), 0, 2) * 2
    x2 = x2_ + (jnp.floor(x1) % 2)
    y = jnp.concatenate([x1[:, None], x2[:, None]], 1) * 2
    return y

def create_standardized_dataset(dataset):
    mean = jnp.mean(dataset, axis=0)
    std = jnp.std(dataset, axis=0) + 1e-6
    dataset = (dataset - mean) / std
    return dataset, mean, std


def sample_from_checkerboard_standardized(key, dataset_size):
    y = sample_from_checkerboard(key, dataset_size)
    mean = CHECKERBOARD_DATASET_MEAN
    std = CHECKERBOARD_DATASET_STD
    y = (y-mean)/std
    return y

### nnrg toy dataset

LATTICE_SIZE = 8
"""must be a power of 2"""
TOY_LATTICE_DATASET_MEAN = jnp.zeros(LATTICE_SIZE)
TOY_LATTICE_DATASET_STD = jnp.array([4/jnp.sqrt(3)]*LATTICE_SIZE)

std = jnp.array([4/jnp.sqrt(3)]*LATTICE_SIZE)
def sample_from_checkervector_dataset(key, dataset_size):
    """generates a toy dataset for use in testing the neural RG implementation. The output array is of shape
    (data_set_size, `LATTICE_SIZE`)
    """
    key = jr.split(key, LATTICE_SIZE//2)
    checkerboard_samples = jax.vmap(lambda key, dataset_size: sample_from_checkerboard(key, dataset_size), in_axes=(0, None))(key, dataset_size)
    dataset = jnp.concatenate(checkerboard_samples, axis=1)
    return jnp.roll(dataset, 1, axis=1)

def sample_from_checkervector_dataset_standardized(key, dataset_size):
    """Performs the same operation as `sample_from_checkervector_dataset` except the datapoints have been standardized"""
    y = sample_from_checkervector_dataset(key, dataset_size)
    return (y-TOY_LATTICE_DATASET_MEAN)/TOY_LATTICE_DATASET_STD


#### Low temp

def get_K_alpha(L, T):
  adj = jnp.zeros((L,L)) # symmetric adjacency
  # the ith spin is adjacent to the i+1th and the i-1th spin
  adj = jnp.eye(L, k=1) + jnp.eye(L, k=-1)
  adj = adj.at[0, L-1].set(1)
  adj = adj.at[L-1, 0].set(1)

  K = adj/T
  w, _ = jnp.linalg.eigh(K)
  alpha = 0.1-w.min()
  return K, alpha

def transform_with_continuous_relaxation(key, configurations):

    return jr.uniform(key, (configurations.shape[0],configurations.shape[1])) * configurations + 0.5*configurations


def sample_from_low_temp_ising(key, dataset_size, lattice_size):
    dataset = jnp.zeros((dataset_size, lattice_size))
    selectors = jr.bernoulli(key, shape=(dataset_size,))
    return selectors[:, None].astype(jnp.float32) * jnp.ones((dataset_size, lattice_size))


#%matplotlib inline
import jax.numpy as jnp # TODO: JIT EVERYTHING
import jax

from math import pi
import matplotlib.pyplot as plt
# change some of the defaults for plots
plt.rcParams['text.usetex'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'] = [18,6]
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
from IPython.display import display, Markdown, Latex, Math, Pretty

from jax import random as jr

L_test = 100          # lattice size is L
J_test = 1.0          # gauge coupling parameter
kB_test = 1.0         # the Boltzman constant

import jax
import jax.numpy as jnp
from jax import random as jr

def metropolis_single_chain(key, s, T, kB):
    '''
    Runs the Metropolis algorithm for a single chain for one unit of
    Monte Carlo time (MCT), using JAX-friendly control flow primitives.
    '''
    lattice_size = s.shape[0]
    oldE = energy(s, J_test)

    def step_fn(carry, _):
        key, s, oldE = carry
        key, subkey_randint, subkey_rand = jr.split(key, 3)

        # 1. Pick a random spin to flip
        i = jr.randint(subkey_randint, (), 0, lattice_size)
        s_flipped = s.at[i].set(s[i] * -1)

        # 2. Calculate energy difference
        newE = energy(s_flipped, J_test)
        deltaE = newE - oldE

        # 3. Metropolis acceptance criterion:
        # If deltaE <= 0, jnp.maximum is 0.0, exp(0) = 1.0 -> always accept (uniform rand is always < 1.0)
        # If deltaE > 0, evaluates normally to exp(-deltaE / k_B T) without overflow risk
        accept = jr.uniform(subkey_rand) < jnp.exp(-jnp.maximum(deltaE, 0.0) / (kB * T))

        # 4. Update the state without using Python if/else statements
        s = jnp.where(accept, s_flipped, s)
        oldE = jnp.where(accept, newE, oldE)

        return (key, s, oldE), None

    init_carry = (key, s, oldE)

    # jax.lax.scan operates like an efficient, compiled loop
    (final_key, final_s, _), _ = jax.lax.scan(step_fn, init_carry, None, length=lattice_size)
    return final_s


# --- MULTI-CHAIN VECTORIZATION ---
# jax.vmap automatically adds a batch dimension to run multiple chains in parallel.
# in_axes=(0, 0, None, None) indicates that:
#   - key and s have a leading dimension corresponding to each chain (axis 0)
#   - T and kB are shared global scalars across all chains
metropolis = jax.vmap(metropolis_single_chain, in_axes=(0, 0, None, None))

'''
Functions to calculate Energy (E) and Magnetic Moment (M) of the L*L spin lattice
Particles at the edge of the lattice rollover for adjacent calculation using periodic boundary conditions through np.roll
'''

def energy( s, coupling) :
    '''!Returns the energy divided by the number of sites of the configuration `s` , assuming that the configuration is from a
    1D Ising model with periodic boundary conditions and that is subject to no external magnetic field'''
    # this is the energy for each site
    E = -coupling * ( s * jnp.roll( s, 1 ) )
    # and this is the avg energy per site
    return jnp.sum( E ) / s.size

#Simple sum over spin of all particles
def magnetization( s ) :
    '''Returns the magnetization per spin of the configuration `s`, assuming that the configuration is from a
    1D Ising model with periodic boundary conditions and that is subject to no external magnetic field'''
    return jnp.sum( s ) / s.size

#Creates an LxL lattice of random integer spins with probability (p) to be +1 (and 1-p to be -1)
def randomLattice(key, L, p ) :

    return ( jr.uniform(key, L ) < p ) * 2 - 1
def magnetization(configuration):
    '''Returns the microscopic-state magnetization of the 1D Ising model `configuration` (assuming that it is not subject to
    an external magnetic field) which is defined as the sum of the configuraton's spins'''
    return jnp.sum( configuration )






from numpyro.diagnostics import effective_sample_size

def get_help_finding_int_time(T, L, kB, J, n=1000, p=0.5):
    seed = 170
    key= jr.PRNGKey(1)
    key, key_lattice_gen = jr.split(key, 2)
    spinLattice = randomLattice(key_lattice_gen, L, p)

    # 1. Define the loop body for jax.lax.scan
    def scan_body(carry, _):
        key, lattice = carry
        key, subkey = jr.split(key, 2)

        # Evolve the lattice using your existing metropolis function
        lattice = metropolis_single_chain(subkey, lattice, T, kB)

        # Compute observables for the current step
        M_i = magnetization(lattice)
        E_i = energy(lattice, J)

        return (key, lattice), (M_i, E_i)

    # 2. Use jax.jit to compile the entire loop into a single XLA operation
    @jax.jit
    def run_simulation(key, spinLattice):
        _, (M, E) = jax.lax.scan(scan_body, (key, spinLattice), None, length=n)
        return M, E

    # 3. Run the compiled simulation
    M, E = run_simulation(key, spinLattice)

    # Calculate effective sample size and integrated time
    num_samples_effective = effective_sample_size(M[None, :])
    num_samples = n
    numpyro_integrated_time = num_samples / num_samples_effective
    print("numpyro integrated: " + str(numpyro_integrated_time))

    return int(math.ceil(float(numpyro_integrated_time)))



def get_1D_ising_configs_from_metropolis(key, n, T, L, J, kB):
    p = 0.5#:)                             # probability for the initial random lattice
    configs = []

    key, key_lattice_gen = jr.split(key, 2)
    spinLattice = randomLattice(key_lattice_gen, L, p )

    #Run metropolis algo for N time steps and record energy, magnetic moments of each random lattice config
    metro = lambda key, s, T: metropolis_single_chain(key, s, T, kB)
    for i in range( n ) :
        key, subkey = jr.split(key, 2)
        spinLattice = metro(subkey, spinLattice, T )
        configs.append(spinLattice)

    return configs

def get_1D_ising_samples_discrete_from_metropolis_output(metropolis_output, integrated_time, burn_in):
    return metropolis_output[burn_in::integrated_time]




def sample_continuous_from_discrete(key, discrete: jax.Array, K, alpha, lattice_size):
    cov = (K + alpha*jnp.eye(lattice_size))
    return jr.multivariate_normal(key, (K + alpha*jnp.eye(lattice_size))@discrete, cov)

def sample_continuous_dataset_from_discrete(key, discrete_dataset):
    pass

def sample_discrete_configurations(key, dataset_size, lattice_size, temp, integrated_time, burn_in, num_chains):
    """
    Samples Ising configurations using multiple chains in parallel.

    Args:
        num_chains: Number of parallel chains to run.
    """
    assert dataset_size % num_chains == 0
    total_steps = int((dataset_size * jnp.ceil(integrated_time) // num_chains)) + burn_in
    print(total_steps)

    # 1. Initialize multiple chains: (num_chains, lattice_size)
    keys = jr.split(key, num_chains)
    initial_lattices = jax.vmap(randomLattice, in_axes=(0, None, None))(keys, lattice_size, 0.5)

    # 2. Run Metropolis across all chains
    # We need to maintain the state across iterations
    def loop_body(carry, _):
        key, s = carry
        key, subkey = jr.split(key)
        # metropolis is the vmapped function defined in your script
        new_s = metropolis(jr.split(subkey, num_chains), s, temp, 1.0)
        return (key, new_s), new_s

    # Run for total_steps
    # Using scan for JAX-friendly efficient looping
    _, all_configs = jax.lax.scan(loop_body, (key, initial_lattices), None, length=total_steps)

    # all_configs shape: (total_steps, num_chains, lattice_size)
    # 3. Apply burn-in and integrated_time sampling
    # Flattening chains into a single stream of samples
    flattened_configs = all_configs[burn_in::integrated_time].reshape(-1, lattice_size)

    # Return requested number of samples
    return flattened_configs

def sample_from_continuous_relaxation_1D(key, dataset_size, lattice_size, temp, integrated_time, burn_in, num_chains):
    discrete_key, key_continuous = jr.split(key, 2)
    dataset_discrete = sample_discrete_configurations(discrete_key, dataset_size, lattice_size, temp, integrated_time, burn_in, num_chains)
    keys_continuous = jr.split(key_continuous, dataset_size)
    K, alpha = get_K_alpha(lattice_size, temp)
    dataset_continuous = jax.vmap(sample_continuous_from_discrete, (0, 0, None, None, None))(keys_continuous, dataset_discrete, K, alpha, lattice_size)
    return dataset_continuous


### High temp ising



def ising_model(L, J, beta):
    """
    source: Gemini
    L: Lattice size L
    J: Coupling constant (positive for ferromagnetic)
    beta: Inverse temperature (1/(kB * T))
    """
    # 1. Define spins as Bernoulli (0/1).
    # We transform them to +/-1 for the energy calculation.
    # We use a plate for each dimension of the lattice.
    with numpyro.plate("cols", L):
        # Sampling 0/1 spins
        spins_01 = numpyro.sample("spins", dist.Bernoulli(0.5))
        spins = 2 * spins_01 - 1  # Convert to +/- 1

    # 2. Calculate the Hamiltonian (Energy)
    # Nearest neighbor interactions (with periodic boundary conditions)
    interaction = jnp.sum(spins * jnp.roll(spins, shift=1, axis=0))

    energy = -J * (interaction)

    # 3. Use numpyro.factor to add log P(s) = -beta * Energy to the model
    # Note: NumPyro's factor adds to the log-joint density.
    numpyro.factor("energy_factor", -beta * energy)

def magnetization(configuration):
    '''Returns the microscopic-state magnetization of the 1D Ising model `configuration` (assuming that it is not subject to
    an external magnetic field) which is defined as the sum of the configuraton's spins'''
    return jnp.sum( configuration )

# i need an ensemble of samplers
RECOMMENDED_R_HAT_THRESHOLD = 1.1
kB = 1.0         # the Boltzman constant
class IsingModel:

    WARM_UP = 1000 # TODO: Watch out for equilibriation period
    """length of the warm-up period"""
    CONVERGENCE_CHECK_PERIOD_LENGTH = 1000
    """number of samples to used in determining if equilibration has been reached i.e. length of the
    chain used to compute R-hat (note: the chain used to compute R-hat does not include samples that were generated
    during the warm-up perio)"""
    CALIBRATION_PERIOD = 1000
    """number of mcmc steps used to determine the integrated time, after the warm-up period"""
    NUM_CHAINS = 4
    """number of mcmc chains used in computing the integrated time"""
    L: int
    J: float
    beta: float
    integrated_time: int
    """integrated time for the ising model """


    def __init__(self, key:jr.PRNGKey, L: int, T: float, J: float):
        """

        T:
            temperature
        J:
            coupling parameter"""
        self.kernel = DiscreteHMCGibbs(NUTS(ising_model))
        mcmc = MCMC(self.kernel, num_warmup=self.WARM_UP, num_samples=self.CONVERGENCE_CHECK_PERIOD_LENGTH, num_chains=self.NUM_CHAINS)
        # Assert convergence
        self.L = L
        self.J = J
        self.beta = 1/(kB*T)
        samples = self._sample_ising(key, mcmc)
        magnetization_samples = jax.vmap(jax.vmap(magnetization))(samples)
        self.n_eff = effective_sample_size(magnetization_samples)
        self.integrated_time = int(jnp.ceil(self.CONVERGENCE_CHECK_PERIOD_LENGTH/self.n_eff))
        # DOUBlE CHECK SAMPLE SHAPE
        print(self.integrated_time)
        plt.plot(autocorrelation(magnetization_samples[0]))
        plt.show()
        #

    def _sample_ising(self, key:jr.PRNGKey, mcmc: MCMC):
        """Returns samples of the icing model produced by running `mcmc"""
        mcmc.run(key, self.L, self.J, self.beta)
        return 2*mcmc.get_samples(True)["spins"] -1

    def get_1D_ising_configurations(self, key, dataset_size):
        """Returns an array of shape (`dataset_size`, `self.L) containing independent samples from the 1D Ising model with lattice size `self.L`,
        temperature `self.T`, and coupling parameter `self.J"""
        mcmc = MCMC(self.kernel, num_warmup=self.WARM_UP, num_samples=int(jnp.ceil(self.CONVERGENCE_CHECK_PERIOD_LENGTH/self.n_eff)*dataset_size))
        samples =self._sample_ising(key, mcmc)[0]
        samples = samples[::self.integrated_time]
        # TODO: filter
        return samples

    def get_configurations_with_continuous_relaxation(self, key, dataset_size):
        """Returns a set of 1D ising configurations where up-spins are represented as samples from abs(standard_gaussian) + 1 and where down-spins
        are represented as sampels from -abs(standard_gaussian) - 1"""
        mapping_key, sample_key = jr.split(key, 2)
        samples = self.get_1D_ising_configurations(sample_key, dataset_size)
        return jr.uniform(mapping_key, (dataset_size, self.L)) * samples + 0.5*samples

    # we may find out continuous isisng is beter and have to learn numpyro anyway.
    # is










# ## Train CNF

# %%
BASELINE_LEARNING_RATE = 1e-3

# %% [markdown]
# ### helper functions misc

# %% [code]
CustomData = list[
    tuple[
        Mapping[str, Union[int, float, str, bool]],
        Mapping[str, Union[float, tuple[float, float]]],
        int,
        str,
    ]
]


def make_json_serializable(obj: Any) -> Any:
    """Recursively converts mappings to dicts and tuples to lists

    to ensure the entire structure is completely JSON serializable.
    """
    if isinstance(obj, Mapping):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


# %%
def kinetic_energy_penalty(model, per_submodule_decimator_vector_field_snapshots, per_submodule_disentangler_vf_snapshots, regularization_term_key: jr.PRNGKey):
    """
    Computes a monte carlo estimate of the kinetic-energy penalty:
    Sum over submodules of ∫ 0.5( v(x, t) )^2 dt
    """


    def compute_penalty_single_layer_in_submodule(velocity, key):

        # Evaluate velocity field for this submodule
        integrand = 0.5 * jnp.square(jnp.linalg.norm(velocity, axis=1))

        # Compute integral estimate
        mean_integrand = jnp.mean(integrand)
        return mean_integrand

    def compute_single_submodule_penalty(deci_point_options, disen_point_options, key):
        # Sample time within the specific submodule's interval
        key_dis, key_dec, key_dis_choice, key_deci_choice = jr.split(key, 4)

        penalty_for_disentangler_layer = compute_penalty_single_layer_in_submodule(disen_point_options, key_dis)
        penalty_for_decimator_layer = compute_penalty_single_layer_in_submodule(deci_point_options, key_dec)
        return penalty_for_disentangler_layer + penalty_for_decimator_layer

    # Split the key to ensure unique sampling for each submodule
    num_submodules = len(model.nnrg.submodules)
    keys = jr.split(regularization_term_key, num_submodules)

    # FIX: Use a list comprehension to iterate over the Python list of submodules
    # This works perfectly inside jax.jit as it unrolls the loop at compile time.
    penalties = jax.vmap(compute_single_submodule_penalty)(per_submodule_decimator_vector_field_snapshots, per_submodule_disentangler_vf_snapshots, keys)

    return jnp.sum(jnp.array(penalties))

def penalties_on_test_data(model, test_data, loss_key, num_time_samples):
  loss_key, key_ke, key_shots = jr.split(loss_key, 3)
  key_shots = jr.split(key_shots, test_data.shape[0])

  all_coarse, logpp, per_submodule_decimator_vector_field_snapshots, per_submodule_disentangler_vf_snapshots = jax.vmap(lambda m, example, key: llambda(m, example, num_time_samples, key), in_axes=(None, 0, 0))(model, test_data, key_shots)

  keys_ke = jr.split(key_ke, all_coarse.shape[0])

  penalty = jax.vmap(lambda deci_shots, disen_shots, key: jax.checkpoint(kinetic_energy_penalty)(model, deci_shots, disen_shots, key))(per_submodule_decimator_vector_field_snapshots,
                                                                                                                                                            per_submodule_disentangler_vf_snapshots, keys_ke)
  penalty = jnp.mean(penalty)
  return {"ke": penalty, "nll": NLLLoss_2(all_coarse, logpp)}




# %%
def undo_custom_continuous_relaxation(configuration):
    """Returns an array of the same shape where the an element is 1 if the corresponding element in `configuration` is positive and -1
    otherwise"""
    return jnp.where(jnp.array(configuration) > 0, 1, -1)


# %%
from dataclasses import dataclass

@dataclass
class InferenceInfo:
    mean: float
    std: float

# %%
def sample_from_full_nnrg(model, key, lattice_size):
    generation_key, latent_key = jr.split(key, 2)
    return model.generate(generation_key, jr.normal(latent_key, (1,)))[0]

# %%
def get_depth_of_full_nnrg(lattice_size):
  return jnp.log2(lattice_size)

# %%
def handy_function(diff_tensor, denom_term):
    """
    Operates on a tensor of differences using broadcasting.
    Expects diff_tensor shape: (n, d) for first_sum or (n*n, d) for second_sum.
    """
    # jnp.square and jnp.sum(axis=-1) are faster than vmapping jnp.pow
    sum_sq = jnp.sum(jnp.square(diff_tensor), axis=-1)
    return jnp.exp(-sum_sq / (2 * denom_term))

# Faster implementation using expansion
def analytical_mmd(samples: jax.Array, bandwidth: float):
    n, d = samples.shape
    gamma = 1.0 / (2 * (bandwidth**2))

    # Precompute constants
    bw_sq = jnp.square(bandwidth)
    h_const1 = jnp.pow(jnp.pow(bandwidth, 2)/(2 + jnp.pow(bandwidth, 2)), d/2)
    h_const2 = jnp.pow(jnp.pow(bandwidth, 2)/(1 + jnp.pow(bandwidth, 2)), d/2)

    # 1. Faster first_sum: use jnp.sum on the norm directly
    norm_x = jnp.sum(samples**2, axis=1)
    first_sum = jnp.sum(jnp.exp(-norm_x / (2 * (1 + bandwidth**2))))

   # 2. Optimized second_sum: Broadcasted operation over pairwise differences
    # Use broadcasting to create the diffs: (n, 1, d) - (1, n, d) -> (n, n, d)
    pairwise_diff = samples[:, None, :] - samples[None, :, :]

    # Apply handy_function over the last dimension (d)
    # output shape: (n, n)
    second_sum_matrix = handy_function(pairwise_diff, bw_sq)

    # Remove diagonal (unwanted terms) and sum
    second_sum = second_sum_matrix.sum() - n

    return h_const1 - 2/n * h_const2 * first_sum + 1/(n*(n-1)) * second_sum

# %%
def get_intra_latents(model: WrapperForNNRG, lattice_size, batch_size, key):
    """Returns a (batch_size, lattice_size) array where each row contains the latent variables (not including the final
    coarse-grained variables or the variables provided as input to `model`) that `model` produces while transforming
    coarse-grained variables into a final lattice"""
    regularization_key, latent_generation_key = jr.split(key, 2)
    regularization_key = jr.split(regularization_key, batch_size)
    z = jr.normal(latent_generation_key, (batch_size, lattice_size//(2**len(model.nnrg.submodules))))
    _, intra_latents, _, _ = jax.vmap(model.generate)(regularization_key, z)
    return intra_latents

# %%


# %%
REGULARIZATION_BANDWITH = 0.6
def regularization_on_marginals(model: WrapperForNNRG, data: jax.Array, regularization_key: jr.PRNGKey):
    """Returns the penalty to be added to the loss function such that the marginal distribution of each
    latent variable is gaussia

    data:
        shape is (batch_size, lattice_size) where `lattice_size` is the size of the input"""
    intra_latents = get_intra_latents(model, data.shape[1], data.shape[0], regularization_key)
    penalty = jax.vmap(lambda column: analytical_mmd(column, REGULARIZATION_BANDWITH), in_axes=1)(intra_latents[:, :, None])
    penalty = jnp.where(penalty < 0, 0, penalty)
    penalty = penalty.sum()
    return penalty

# %%
def _regularization_helper_function(intra_latents):
  penalty = jax.vmap(lambda column: analytical_mmd(column, REGULARIZATION_BANDWITH), in_axes=1)(intra_latents[:, :, None])
  penalty = jnp.where(penalty < 0, 0, penalty)
  penalty = penalty.sum()
  return penalty

# %%
def generate_potential_fn(K, alpha, N):
    """
    Generates the potential energy function U(x) for HMC.

    Args:
        K (jnp.ndarray): The N x N symmetric coupling matrix.
        alpha (float): The constant offset.
        N (int): The size of the system.
    """
    # Pre-compute the inverse matrix once
    K_plus_alpha_I = K + alpha * jnp.eye(N)
    inv_K_plus_alpha_I = jnp.linalg.inv(K_plus_alpha_I)

    def potential_energy(x):
        """
        Calculates the potential energy for a given configuration x.
        This is the negative log probability, up to a constant.
        """
        # Calculate the quadratic term: 0.5 * x^T * inv(K + alpha * I) * x
        quadratic_term = 0.5 * jnp.dot(jnp.dot(x.T, inv_K_plus_alpha_I), x)

        # Calculate the log(cosh(xi)) term
        cosh_term = jnp.sum(jnp.log(jnp.cosh(x)))

        # The potential energy is proportional to the negative log likelihood
        return quadratic_term - cosh_term

    return potential_energy



# %%
NLL_COEFF = 1

MARGINAL_REGULARIZATION_COEFF = 6

def llambda(model, data, num_time_samples, key):
  module_output = model.inference(data, num_time_samples, key)
  return module_output[COARSE_VAR_NAME], module_output[LOGP_NAME], module_output[VECTOR_FIELD_SNAPSHOT_NAME][DECIMATOR_CNF_NAME], module_output[VECTOR_FIELD_SNAPSHOT_NAME][DISENTANGLER_CNF_NAME]

def NLLLoss_2(latent_variables, log_likelihood):
  print(latent_variables.shape)
  log_likelihood += jax.vmap(normal_log_likelihood)(latent_variables)
  return -jnp.mean(log_likelihood)

def NLLLoss(model, data):
    """
    data:
        shape (batch_size, lattice size)
    """

    latent_variables, log_likelihood = jax.vmap(llambda)(data)
    log_likelihood += jax.vmap(normal_log_likelihood)(latent_variables)
    return -jnp.mean(log_likelihood)

# %%
def nll(inference_info, data, loss_key):
    model = inference_info.model
    data = (data-inference_info.mean)/inference_info.std
    train_key = jr.split(loss_key, data.shape[0])

    latent_variables, log_likelihood = jax.vmap(lambda data, key: model.sample_and_compute_density(data, key=key, is_forward_direction=False))(data, key=train_key)
    log_likelihood += jax.vmap(normal_log_likelihood)(latent_variables)
    return float(-jnp.mean(log_likelihood))  # minimise negative log-likelihood

# %%
def nll_nnrg(model: WrapperForNNRGSubModule, inference_info: InferenceInfo, data: jax.Array, loss_key: jr.PRNGKey):
    """Computes the mean NLL of that `model` achieves on the dataset `data`

    `data`:
        shape (batch_size x lattice_size) where `lattice_size` is the size of vector that `model` expects as input
    inference_info:
        specifies the mean and std that should be used to transform `data` before providing it as input to `model` (transformed data = (data- mean)/std)
        (for use if the data from the target distribution was standardized before being provided to the model during training)
    """
    data = (data-inference_info.mean)/inference_info.std
    return NLLLoss(model, data)

# %%
def sample_from_nnrg(model: WrapperForNNRGSubModule, key: jr.PRNGKey, lattice_size:int) -> jax.Array:
    """Returns a sample from the neural network
    renormalization group `model`
    if called multiple times with a different key each time, the result is multiple independent samples from the neural network
    renormalization group `model`"""
    other_collective_variable_key, collective_variable_key = jr.split(key, 2)
    return model.generate(other_collective_variable_key, jr.normal(collective_variable_key, (lattice_size//2)))[0]

# %%
class ModelInferenceInfo:
    def __init__(self, model, mean, std):
        self.model  = model
        self.mean = mean
        self.std = std

# %%


# %%
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np # NumPy is imported to aid in converting the JAX array efficiently

def plot_jax_rows_as_images(jax_array_2d):
    """

    Takes a 2D JAX array and creates a matplotlib figure where each row
    in the array is rendered as an image in a subplot.

    Args:
        jax_array_2d: A 2D jax.numpy.ndarray.
    """
    num_rows, num_cols = jax_array_2d.shape

    # Determine the number of rows/columns for the subplot grid.
    # For a vertical stack, use num_rows rows and 1 column.
    fig, axes = plt.subplots(num_rows, 1, figsize=(5, num_rows * 5))

    # Adjust axes if only one row exists (subplots returns a single axis object, not an array)
    if num_rows == 1:
        axes = [axes]

    # Iterate over each row and plot it as an image in the corresponding subplot
    for i in range(num_rows):
        # Convert the JAX array row to a NumPy array for efficient plotting
        # Matplotlib works best with NumPy arrays
        row_data = np.array(jax_array_2d[i, :])

        # Display the row as an image
        # Use 'gray' colormap for general 2D data visualization
        im = axes[i].imshow(row_data.reshape(1, -1), cmap='viridis', aspect='auto')

        # Add a color bar for reference
        plt.colorbar(im, ax=axes[i])

        # Set title and remove axis ticks for cleaner image presentation
        axes[i].set_title(f'Row {i+1} as Image')
        axes[i].axis('off')

    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    plt.show()
"""
# Example Usage:
# 1. Create a sample 2D JAX array (e.g., 4 rows, 10 columns of random data)
key = jax.random.PRNGKey(0)
jax_array = jax.random.uniform(key, (4, 10))

# 2. Call the function to plot the array
plot_jax_rows_as_images(jax_array)
"""

# %% [markdown]
### helper functions for visualization

# %% [code]
@jax.jit
def sample_s_given_x(rng_key, x):
    """Samples discrete Ising variables s in {-1, 1}^N given continuous vector x.

    Args:
        rng_key: JAX random PRNGKey.
        x: jnp.ndarray of shape (N,) representing the continuous states.

    Returns:
        jnp.ndarray of shape (N,) with values in {-1.0, 1.0}.
    """
    # Calculate the probability that s_i = +1, which is equivalent to 1 / (1 + e^(-2*x))
    p_positive = jax.nn.sigmoid(2.0 * x)

    # Generate uniform random values between 0 and 1
    u = jax.random.uniform(rng_key, shape=x.shape)

    # Assign 1.0 if u < p_positive, otherwise assign -1.0
    s = jnp.where(u < p_positive, 1.0, -1.0)

    return s

def get_discrete_samples_from_model(model, key, lattice_size, num_samples):
  key_continuous, key_discrete = jr.split(key)
  samples_from_model = jax.vmap(lambda key: sample_from_full_nnrg(model, key, lattice_size))(jr.split(key_continuous, num_samples))
  discrete_samples = jax.vmap(lambda sample, key: sample_s_given_x(key, sample))(samples_from_model, jr.split(key_discrete, num_samples))
  return discrete_samples


# %% [code]
def get_discrete_samples(continuous_samples, key):
  key_discrete = jr.split(key, continuous_samples.shape[0])
  discrete_samples = jax.vmap(lambda sample, key: sample_s_given_x(key, sample))(continuous_samples, key_discrete)
  return discrete_samples


# %

# %% [code]
"""
Visualize a batch of 1D Ising spin configurations arranged in a grid.

Input:
    spins: jax.numpy array of shape (batch, N), entries in {-1, +1}
           (batch = number of spin configurations, N = chain length)

Each configuration is drawn as a 1D strip (black = -1, white = +1),
and the strips are tiled into a grid, one per batch element.
"""

import math
import jax.numpy as jnp
import matplotlib.pyplot as plt


def visualize_spin_batch(spins, ncols=None, cell_size=1.2, cmap="Greys"):
    """
    Arrange each batch element's 1D spin configuration into a grid of subplots.

    Args:
        spins: (batch, N) array-like of +-1 spins (jax or numpy array).
        ncols: number of grid columns. Defaults to ceil(sqrt(batch)).
        cell_size: size (inches) of each subplot cell.
        cmap: matplotlib colormap used to render the spin strip.

    Returns:
        (fig, axes) the matplotlib figure and axes array.
    """
    spins = jnp.asarray(spins)
    if spins.ndim != 2:
        raise ValueError(f"expected a 2D array (batch, N), got shape {spins.shape}")

    batch, n = spins.shape

    if ncols is None:
        ncols = math.ceil(math.sqrt(batch))
    nrows = math.ceil(batch / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * cell_size, nrows * cell_size * 0.4),
        squeeze=False,
    )

    # Convert once to a plain numpy-like structure for plotting
    spins_np = jnp.asarray(spins)

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        if idx < batch:
            config = spins_np[idx].reshape(1, n)  # reshape to a 1-row strip
            ax.imshow(config, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
            ax.set_title(f"#{idx}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig, axes

# %% [code]
def compare_model_vs_validation(
    model_samples,
    val_samples,
    ncols=None,
    cell_size=1.2,
    cmap="Greys",
    n_show=None,
):
    """
    Visualize samples from a generative model side-by-side with validation
    samples, plus a couple of summary statistics for a quick sanity check.

    Args:
        model_samples: (batch_m, N) array of +-1 spins from the generative model.
        val_samples:   (batch_v, N) array of +-1 spins from the validation set.
        ncols: columns per grid (applied to both grids independently).
        cell_size: size (inches) of each subplot cell.
        cmap: colormap for the spin strips.
        n_show: if given, only the first n_show samples of each set are plotted
                (statistics are still computed on the full arrays passed in).

    Returns:
        fig: matplotlib figure with two grids (model on top, validation below)
             and a text panel of summary statistics.
        stats: dict with per-set mean magnetization, magnetization std,
               and mean nearest-neighbor correlation <s_i s_{i+1}>.
    """
    model_samples = jnp.asarray(model_samples)
    val_samples = jnp.asarray(val_samples)

    if model_samples.ndim != 2 or val_samples.ndim != 2:
        raise ValueError("both model_samples and val_samples must be 2D (batch, N)")
    if model_samples.shape[1] != val_samples.shape[1]:
        raise ValueError(
            f"chain length mismatch: model N={model_samples.shape[1]}, "
            f"val N={val_samples.shape[1]}"
        )

    def _stats(spins):
        magnetization = jnp.sum(spins, axis=1)  # per-sample magnetization
        nn_corr = jnp.mean(spins[:, :-1] * spins[:, 1:])  # <s_i s_{i+1}>, averaged
        return {
            "mean_magnetization": float(jnp.mean(magnetization)),
            "std_magnetization": float(jnp.std(magnetization)),
            "mean_nn_correlation": float(nn_corr),
        }

    stats = {
        "model": _stats(model_samples),
        "validation": _stats(val_samples),
    }

    model_plot = model_samples if n_show is None else model_samples[:n_show]
    val_plot = val_samples if n_show is None else val_samples[:n_show]

    if ncols is None:
        ncols = math.ceil(math.sqrt(max(model_plot.shape[0], val_plot.shape[0])))

    def _grid_rows(n_needed):
        return math.ceil(n_needed / ncols)

    rows_model = _grid_rows(model_plot.shape[0])
    rows_val = _grid_rows(val_plot.shape[0])
    total_rows = rows_model + rows_val

    fig, axes = plt.subplots(
        total_rows, ncols,
        figsize=(ncols * cell_size, total_rows * cell_size * 0.4 + 0.6),
        squeeze=False,
    )

    def _fill(axes_block, data, label):
        n = data.shape[0]
        nrows_block = axes_block.shape[0]
        for idx in range(nrows_block * ncols):
            r, c = divmod(idx, ncols)
            ax = axes_block[r][c]
            if idx < n:
                strip = data[idx].reshape(1, -1)
                ax.imshow(strip, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
                if idx == 0:
                    ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center")
            ax.set_xticks([])
            ax.set_yticks([])

    _fill(axes[:rows_model], model_plot, "model")
    _fill(axes[rows_model:], val_plot, "validation")

    stats_txt = (
        f"model:      <m>={stats['model']['mean_magnetization']:.3f}  "
        f"std(m)={stats['model']['std_magnetization']:.3f}  "
        f"<s_i s_i+1>={stats['model']['mean_nn_correlation']:.3f}\n"
        f"validation: <m>={stats['validation']['mean_magnetization']:.3f}  "
        f"std(m)={stats['validation']['std_magnetization']:.3f}  "
        f"<s_i s_i+1>={stats['validation']['mean_nn_correlation']:.3f}"
    )
    fig.suptitle(stats_txt, fontsize=8, family="monospace", y=1.02, ha="center")
    fig.tight_layout()

    return fig, stats


#
# %% [markdown]
# ### helper functions for training

# %%
from flax.training import orbax_utils
from orbax.checkpoint import v1 as ocp


# %%
SYMMETRY_REGULARIZATION_COEFF = 1

# %%
class KESchedule:
  ALPHA: int


  def __init__(self, initial_val, num_steps_till_zero):
    self.coeff = initial_val
    self.ALPHA = 1
    self.num_steps_till_0 = num_steps_till_zero

  def get_next(self, step):
    return jnp.max(jnp.array([self.coeff * (1-(step)/(self.num_steps_till_0)), 0])) # try enough to change number of epochs to maximum half of patience, and minimum 3 epochs




# %%
def choose_relevant_sample_to_use_in_KE_penalty(configuration):
  return configuration[:2] # this seems really bad

PATIENCE_NUM_EPOCHS = 75

def get_model_file_identifier(lr: float,
               ke_schedule: KESchedule,
               coeff_marginal_regularization: float,
               coeff_main_loss_term: float,
               num_time_samples,
               num_time_samples_test,
               steps=10000,
               check_for_overfit_every=100,
               desc=""):
  return f"m{lr}{steps}{ke_schedule.coeff}_marg{coeff_marginal_regularization}_main{coeff_main_loss_term}_ntime{num_time_samples}_{num_time_samples_test}_overfitCheck{check_for_overfit_every}_patience{PATIENCE_NUM_EPOCHS}_{desc}test"



def get_model_file_name(
               lr: float,
               ke_schedule: KESchedule,
               coeff_marginal_regularization: float,
               coeff_main_loss_term: float,
               num_time_samples,
               num_time_samples_test,
               steps=10000,
               check_for_overfit_every=100,
               desc=""):
    return get_model_file_identifier(lr, ke_schedule, coeff_marginal_regularization, coeff_main_loss_term, num_time_samples, num_time_samples_test, steps, check_for_overfit_every, desc) + ".eqx"



def train_nnrg(model: WrapperForNNRGSubModule,
               dataloader: StreamingDataLoader,
               loss_key,
               lr: float,
               ke_penalty_coeff: float,
               coeff_marginal_regularization: float,
               coeff_main_loss_term: float,
               num_time_samples,
               dataset_test,
               num_time_samples_test,
               ke_schedule: KESchedule,
               steps=10000,
               exact_logp=True,
               weight_decay=1e-5,
               print_every=100,
               check_for_overfit_every=100,
               desc="",
                save_every=1500):
  optim = optax.adamw(lr, weight_decay=weight_decay)
  opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))


  tracker = OverfitTracker(patience=PATIENCE_NUM_EPOCHS*dataloader.array.shape[0]/dataloader.batch_size/check_for_overfit_every, min_delta=0.01)

  fname = get_model_file_name(lr, ke_schedule, coeff_marginal_regularization, coeff_main_loss_term, num_time_samples, num_time_samples_test, steps, check_for_overfit_every, desc)
  opt_state_fname = f"m{lr}{steps}_test.eqx"
  pth = siren_model_dir + fname
  pth_opt_state = siren_model_dir + opt_state_fname


  NUM_TIME_SAMPLES = 40

  @eqx.filter_value_and_grad(has_aux=True)
  def loss(model, data, loss_key):
    loss_key, key_ke, key_shots = jr.split(loss_key, 3)
    key_shots = jr.split(key_shots, data.shape[0])

    all_coarse, logpp, per_submodule_decimator_vector_field_snapshots, per_submodule_disentangler_vf_snapshots = jax.vmap(lambda m, example, key: llambda(m, example, num_time_samples, key), in_axes=(None, 0, 0))(model, data, key_shots)

    keys_ke = jr.split(key_ke, all_coarse.shape[0])

    penalty = jax.vmap(lambda deci_shots, disen_shots, key: jax.checkpoint(kinetic_energy_penalty)(model, deci_shots, disen_shots, key))(per_submodule_decimator_vector_field_snapshots,
                                                                                                                                                              per_submodule_disentangler_vf_snapshots, keys_ke)
    penalty = jnp.mean(penalty)

    marginal_regularization_penalty = regularization_on_marginals(model, data, loss_key)
    main_loss = NLLLoss_2(all_coarse, logpp)
    total_loss = coeff_main_loss_term*main_loss + coeff_marginal_regularization*marginal_regularization_penalty + ke_schedule.get_next(step)*penalty # optimization improvement: lamdba within jit
    return total_loss, (penalty, marginal_regularization_penalty, main_loss)

  @eqx.filter_jit
  def validation_loss(model, loss_key):
    key_shots_val, key_val = jr.split(loss_key, 2)
    key_shots_val = jr.split(key_shots_val, dataset_test.shape[0])
    all_coarse_val, logpp_val = jax.vmap(lambda m, example, key: llambda(m, example, num_time_samples_test, key), in_axes=(None, 0, 0))(model, dataset_test, key_shots_val)[:2]
    val_loss = NLLLoss_2(all_coarse_val, logpp_val)
    return val_loss


  @eqx.filter_jit
  def make_step(model: WrapperForNNRGSubModule, opt_state, data, loss_key):

      (value, penalties), grads = loss(model, data, loss_key)
      loss_key = jr.split(loss_key, 1)[0]
      updates, opt_state = optim.update(
          grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
      )
      model = eqx.apply_updates(model, updates)
      return value, penalties, model, opt_state, loss_key

  step = 0
  best_model = None
  best_loss = float('inf')
  loss_msgs = []
  ke_penalty_of_best_model = float('inf')
  loss_key, key_val = jr.split(loss_key, 2)
  overfitting = False
  while step < steps:
      val_loss = None
      start = time.time()

      data = dataloader(step)
      step = step + 1
      value, (ke_penalty, penalty_marginal_distribution, main_loss), model, opt_state, loss_key = make_step(
          model, opt_state, data, loss_key
      )

      end = time.time()
      if (step % check_for_overfit_every == 0) or steps == steps-1:
        val_loss = validation_loss(model, key_val)
        key_val = jr.fold_in(key_val, step)
        tracker_verdict = tracker.update(val_loss)
        if tracker_verdict == "stop":
          overfitting = True
          print(f"Overfitting! val loss: {val_loss}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            ke_penalty_of_best_model = ke_penalty
      if (step % print_every) == 0 or step == steps - 1:
          loss_msg = f"Step: {step}, Loss: {value}, KE Penalty: {ke_penalty}, Marg Penalty: {penalty_marginal_distribution}, just NLL: {main_loss}, Val loss: {val_loss}, Computation time: {end - start}"
          print(loss_msg)
          loss_msgs.append(loss_msg)
      if (step % save_every) == 0 or step == steps - 1 or step == steps or step == 1:
        nrg_wrapper_saver(pth, {"depth": len(model.nnrg.submodules)}, best_model)
      if overfitting and ke_schedule.get_next(step) == 0:
        break

  return best_model, (opt_state, loss_msgs, ke_penalty_of_best_model)


# %%
def train_nnrg_ising(model: WrapperForNNRGSubModule,
               dataloader: StreamingDataLoader,
               loss_key,
               lr: float,
               steps=10000,
               exact_logp=True,
               weight_decay=1e-5,
               print_every=100,
               desc="",
                     T=2.0,
                     save_every=1500):
    """
    desc:
        brief description of this training"""

    step_size = steps*2//3
    gamma = 0.7

    # Create the learning rate schedule
    # optax.exponential_decay provides a simple way to implement this step-wise gamma decay.
    scheduler = optax.exponential_decay(
    init_value=lr,
    transition_steps=step_size,
    decay_rate=gamma,
    # Setting the staircase parameter to True makes the decay step-wise,
    # matching the PyTorch StepLR behavior.
    staircase=True
)

    optim = optax.adamw(learning_rate=scheduler, weight_decay=weight_decay)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    K, alpha = get_K_alpha(dataloader.data_size, T)
    M = K + alpha * jnp.eye(dataloader.data_size)
    upper_bound_on_component_variance = jnp.max(jnp.diag(M))
    upper_bound_on_component_std = jnp.sqrt(upper_bound_on_component_variance)
    potential_energy = generate_potential_fn(K, alpha, dataloader.data_size)

    fname = f"m{lr}{steps}{T}_test" + ".eqx"
    opt_state_fname = f"m{lr}{steps}{T}.eqx"
    pth = siren_model_dir + fname
    pth_opt_state = siren_model_dir + opt_state_fname

    def PDD_and_regularization(model, lattice_size, batch_size, loss_key):
      latent_key, model_key = jr.split(loss_key, 2)
      latent_key = jr.split(latent_key, batch_size)
      model_key = jr.split(model_key, batch_size)
      z = jax.vmap(lambda key: jr.normal(key, (1,)))(latent_key)
      generated, intra_latent, log_likelihood_partial, delta_log_likelihood =jax.vmap(lambda key, latent: model.generate(key, latent))(model_key, z)
      log_likelihood = jax.vmap(normal_log_likelihood)(z)+log_likelihood_partial - delta_log_likelihood
      suppression_of_tendancy_to_collapse_to_one_mode = 0 # this variable can be used to implement the symmetry regularization mentioned in Appendix D
      main_loss_term = jnp.mean(log_likelihood + jax.vmap(potential_energy)(generated*upper_bound_on_component_std))
      regularization_penalty = MARGINAL_REGULARIZATION_COEFF*_regularization_helper_function(intra_latent)
      return main_loss_term, regularization_penalty, suppression_of_tendancy_to_collapse_to_one_mode

    @eqx.filter_value_and_grad(has_aux=True)
    def loss(model, data, loss_key):
        main_loss_term, regularization_penalty, symmetry_regularization = PDD_and_regularization(model, dataloader.data_size, dataloader.batch_size, loss_key)
        total_loss = main_loss_term + regularization_penalty #+ SYMMETRY_REGULARIZATION_COEFF*symmetry_regularization
        return total_loss, regularization_penalty

    @eqx.filter_jit
    def make_step(model: WrapperForNNRGSubModule, opt_state, data, loss_key):

        (total_loss, regularization_penalty), grads = loss(model, data, loss_key)
        loss_key = jr.split(loss_key, 1)[0]
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return total_loss, model, opt_state, loss_key, regularization_penalty

    step = 0
    best_model = None
    best_loss = float('inf')
    while step < steps:
        start = time.time()

        data = dataloader(step)
        step = step + 1
        value, model, opt_state, loss_key, regularization_penalty = make_step(
            model, opt_state, data, loss_key
        )

        end = time.time()
        if (step % print_every) == 0 or step == steps - 1:
            if value < best_loss:
                best_loss = value
                best_model = model
            print(f"Step: {step}, Loss: {value}, Reg Penalty: {regularization_penalty}, Computation time: {end - start}")
        if (step % save_every) == 0 or step == steps - 1 or step == steps or step == 1:

          eqx.tree_serialise_leaves(pth_opt_state, {"step": step, "opt": opt_state})
          nrg_wrapper_saver(pth, {"depth": len(model.nnrg.submodules)}, best_model)

    return best_model, opt_state

# %%
def create_visualizations_nnrg(inference_info: InferenceInfo, sample_from_model: Callable[[jr.PRNGKey], jax.Array ], out_path: str, dataset: jax.Array, width: int, height: int, sample_key: jr.PRNGKey):
    """Plots samples from an Neural Network Renormalization group against samples from the target dataset

    # TODO: allow, to be provided as input, which variables in the lattice you want
    inference_info:
        specifies the mean and std that should be used to transform outputs of the Neural Network Renormalization group model (transformed output = raw_output * std + mean)
        (for use if the data from the target distribution was standardized before being provided to the model during training)
    sample_from_model:
        returns a single sample from the Neural Network Renormalization Group model
    out_path:
        where to store the resulting figure
    dataset:
        contains samples from the target distribution"""
    out_path = pathlib.Path(out_path)

    mean = inference_info.mean
    std = inference_info.std
    num_samples = dataset.shape[0]
    sample_key = jr.split(sample_key, num_samples)
    samples = jax.vmap(sample_from_model)(key=sample_key) # FIX

    samples = samples * std + mean
    x = samples[:, 1]
    y = samples[:, 2]

    total_cols = 2

    fig = plt.figure(
        figsize=(total_cols * 6 * height / width, 10)
    )

    gs = fig.add_gridspec(1, total_cols)


    ax_gen = fig.add_subplot(gs[0, -2])
    ax_gen.scatter(x, y, c="black", s=2)
    ax_gen.set_aspect(height / width)
    ax_gen.tick_params(axis='both', which='major', labelsize=20)


    x_coord_true = dataset[:, 0]
    y_coord_true = dataset[:, 1]
    axtrue = fig.add_subplot(gs[0, -1])
    axtrue.scatter(x_coord_true, y_coord_true)
    axtrue.set_aspect(height / width)
    axtrue.tick_params(axis='both', which='major', labelsize=20)


    plt.savefig(out_path)
    plt.show()


# %% [markdown]
# #### for CNF

# %%
def train_on_moons_dataset(
    vector_field: CNFVectorField,
    dataloader,
    loss_key,
    lr=1e-4,
    steps=10000,
    exact_logp=True,
    weight_decay=1e-5,
    penalty_coefficient=0.2,
    print_every=100,
):

    data_size = dataloader(0).shape[1] #--- data_size is 2 (see above)

    model = CNF(
        vector_field_parameterization=vector_field,
        data_size=data_size,
        exact_logp=exact_logp,
    )

    optim = optax.adamw(lr, weight_decay=weight_decay)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    NUM_TIME_SAMPLES = 40
    def kinetic_energy_penalty(model: CNF, data_point: jax.Array, regularization_term_key: jr.PRNGKey):
        """computes a monte carlo estimate of the kinetic-energy inspired penalty term: ∫ 0.5( v(x, t) )^2 dt where x = `data_point` and v() is the velocity field of the CNF `model`"""
        # Evaluate the integrand at randomly sampled times
        interval_start = model.t0
        interval_end = model.t1
        time = jr.uniform(regularization_term_key, (NUM_TIME_SAMPLES,), minval=interval_start, maxval=interval_end)

        velocity = jax.vmap(lambda t, y: model.func(t, y, None), in_axes=(0, None))(time, data_point)
        integrand = 0.5*jnp.pow(velocity, 2)

        #
        mean_integrand = jnp.mean(integrand)
        time_interval_length = interval_end - interval_start
        return time_interval_length*mean_integrand

    @eqx.filter_value_and_grad
    def loss(model, data, loss_key):
        nll_loss_key, regularization_term_key = jr.split(loss_key, 2)
        nll_loss = NLLLoss_and_regularization(model, data, nll_loss_key, False)

        penalty = jax.vmap(lambda model, data_point: kinetic_energy_penalty(model, data_point, regularization_term_key), in_axes=(None, 0))(model, data)
        penalty = jnp.mean(penalty)

        return nll_loss + penalty_coefficient*penalty


    @eqx.filter_jit
    def make_step(model, opt_state, data, loss_key):
        value, grads = loss(model, data, loss_key)

        loss_key = jr.split(loss_key, 1)[0]


        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return value, model, opt_state, loss_key

    step = 0
    while step < steps:
        start = time.time()

        data = dataloader(step)
        step = step + 1
        value, model, opt_state, loss_key = make_step(
            model, opt_state, data, loss_key
        )

        end = time.time()
        if (step % print_every) == 0 or step == steps - 1:
            print(f"Step: {step}, Loss: {value}, Computation time: {end - start}")


    return model, opt_state

# %%
def visualize_vector_field(
    field: Func,
    t: float,
    ax: plt.Axes,
    args=None,
    grid_size=20,
    low=-4.0,
    high=4.0,
):
    """
    Source: ChatGPT
    Visualize a 2D vector field at a fixed time using a quiver plot.

    This function evaluates a VectorField on a regular Cartesian grid
    in R^2 at a specified time t and visualizes the resulting velocity
    vectors using matplotlib's quiver plot. Each arrow represents the
    instantaneous vector field value at its grid location.

    Parameters
    ----------
    field : VectorField
        A callable object implementing the VectorField interface.
        It must accept arguments (t, y, args) and return an array of
        shape (N, 2) representing the vector field evaluated at N points.
    t : float
        The time at which the vector field is evaluated.
    ax: plt.Axes
        the axes on which to place the visualization
    args : Any, optional
        Additional parameters passed directly to the vector field.
        Defaults to None.
    grid_size : int, optional
        Number of grid points per spatial dimension. The total number
        of evaluation points is grid_size^2. Defaults to 20.
    low : float, optional
        Lower bound of the spatial domain for both x and y axes.
        Defaults to -4.0.
    high : float, optional
        Upper bound of the spatial domain for both x and y axes.
        Defaults to 4.0.

    Returns
    -------
    None
        This function produces a matplotlib visualization but does not
        return any values. it also does not call plt.show()"""

    # Create regular grid
    x = jnp.linspace(low, high, grid_size)
    y = jnp.linspace(low, high, grid_size)
    xx, yy = jnp.meshgrid(x, y)

    # Stack grid points into (N, 2)
    points = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

    # Evaluate vector field (batched)
    vector_field_snapshot = jax.vmap(field, in_axes=(None, 0, None))
    vectors = vector_field_snapshot(t, points, args)

    # Convert to numpy for matplotlib
    U = jax.device_get(vectors[:, 0]).reshape(grid_size, grid_size)
    V = jax.device_get(vectors[:, 1]).reshape(grid_size, grid_size)
    X = jax.device_get(xx)
    Y = jax.device_get(yy)

    # Plot
    plt.figure(figsize=(6, 6))
    ax.quiver(X, Y, U, V)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Vector field at t = {t}")
    ax.set_aspect("equal")


# %%


# %%
MANUALLY_IDENTIFIED_BANDWIDTH = 0.06
def create_visualizations(inference_info, out_path, dataset, width, height, sample_key):
    """ Plots a figure containing a) a plot of samples from the provided model and b) a plot of points
    in `dataset`. Also plots a Quiver plot of the CNF vector field at a specific timestep. Saves the figure to the path `out_path`.

    inference_info: ModelInferenceInfo
    out_path: str
        indicates the path to which the figure should be saved
    dataset: jnp.ndarray
        a 2D array where the second dimension is length 2 and the first dimension is batch dimensions"""
    # Plotting code/visualization
    out_path = pathlib.Path(out_path)

    model = inference_info.model
    mean = inference_info.mean
    std = inference_info.std
    num_samples = dataset.shape[0]
    sample_key = jr.split(sample_key, num_samples)
    samples = jax.vmap(model.sample)(key=sample_key)
    sample_flows = jax.vmap(model.sample_flow, out_axes=-1)(key=sample_key)

    samples = samples * std + mean
    x = samples[:, 0]
    y = samples[:, 1]

    n_flow_plots = len(sample_flows)
    num_marginal_plots = 2
    total_cols = n_flow_plots + 2 + num_marginal_plots

    fig = plt.figure(
        figsize=(total_cols * 6 * height / width, 10)
    )

    gs = fig.add_gridspec(2+ num_marginal_plots, total_cols)


    x_resolution = 100
    y_resolution = int(x_resolution * (height / width))
    sample_flows = sample_flows * std[:, None] + mean[:, None]
    x_pos, y_pos = jnp.broadcast_arrays(
        jnp.linspace(jnp.min(x)-1, jnp.max(x) + 1, x_resolution)[:, None],
        jnp.linspace(jnp.min(y), jnp.max(y) + 1, y_resolution)[None, :],
    )
    positions = jnp.stack([jnp.ravel(x_pos), jnp.ravel(y_pos)])
    densities = [stats.gaussian_kde(samples, bw_method=MANUALLY_IDENTIFIED_BANDWIDTH)(positions) for samples in sample_flows]
    for i, density in enumerate(densities):
        ax = fig.add_subplot(gs[0, i])
        density = jnp.reshape(density, (x_resolution, y_resolution))
        ax.imshow(density.T, origin="lower", cmap="plasma")
        ax.set_aspect(height / width)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    NUM_PLOTS_THAT_DONT_DEPICT_FL = 2 + num_marginal_plots
    ax_gen = fig.add_subplot(gs[0, -NUM_PLOTS_THAT_DONT_DEPICT_FL])
    ax_gen.scatter(x, y, c="black", s=2)
    ax_gen.set_aspect(height / width)
    ax_gen.tick_params(axis='both', which='major', labelsize=20)


    x_coord_true = dataset[:, 0]
    y_coord_true = dataset[:, 1]
    axtrue = fig.add_subplot(gs[0, -NUM_PLOTS_THAT_DONT_DEPICT_FL+1])
    axtrue.scatter(x_coord_true, y_coord_true)
    axtrue.set_aspect(height / width)
    axtrue.tick_params(axis='both', which='major', labelsize=20)

    ax_x_dist = fig.add_subplot(gs[0, -2])
    ax_x_dist.hist(x, bins=30, color='skyblue', edgecolor='black')

    ax_y_dist = fig.add_subplot(gs[0, -1])
    ax_y_dist.hist(y, bins=30)


    # -----------------------------
    # Vector field (bottom row)
    # -----------------------------
    ax_vf = fig.add_subplot(gs[1, :])
    visualize_vector_field(model.func, model.t1, ax_vf)

    plt.savefig(out_path)
    plt.show()

# %%


# %%
def evaluate_sample_quality_nnrg(inference_info, dataset, sample_key, lattice_size, bandwidth: float =0.6):
    """Compute the MMD distance between the distribution learned by the model and the target distribution using Gaussian RBF kernel
    of bandwidth `bandwidth`

    inference_info : ModelInferenceInfo
        Contains the trained model and normalization statistics."""
    model = inference_info.model
    mean = inference_info.mean
    std = inference_info.std
    num_samples = dataset.shape[0]
    sample_key = jr.split(sample_key, num_samples)
    samples = jax.vmap(lambda key: sample_from_full_nnrg(model, key, lattice_size))(sample_key)
    samples = samples * std + mean
    x = torch.tensor(np.array(samples), dtype=torch.float32)
    y = torch.tensor(np.array(dataset), dtype=torch.float32)
    # ---------------------------------------------------------
    # 2. Create the MMD metric
    # ---------------------------------------------------------
    mmd_metric = MaximumMeanDiscrepancy(var=bandwidth)  # Gaussian RBF σ² = 1

    mmd_metric.reset()

    # ---------------------------------------------------------
    # 3. Ignite metrics expect batch inputs (x_batch, y_batch).
    #    Here we pass the full datasets as one batch.
    # ---------------------------------------------------------
    mmd_metric.update((x, y))

    # ---------------------------------------------------------
    # 4. Compute the MMD distance
    # ---------------------------------------------------------
    mmd_value = mmd_metric.compute()

    return float(mmd_value)

# %%

# %%
seed = 5678
key = jr.PRNGKey(seed)
model_key, loader_key, loss_key, sample_key, test_key, evaluation_key = jr.split(key, 6)

# %%
class NNRGUnitTestConfig:

    NUM_SAMPLES = 2500
    BATCH_SIZE = 500
    RANDOM_STATE_TRAIN = 0
    RANDOM_STATE_TEST = 2

# %%
class NNRGIsingConfig:

    NUM_SAMPLES = 2500
    BATCH_SIZE = 70
    RANDOM_STATE_TRAIN = 0
    RANDOM_STATE_TEST = 2






def main():
    parser = argparse.ArgumentParser(description='Train NNRG model.')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--ke_penalty_coeff_min', type=float, default=1.0, help='min possible value for KE penalty coefficient')
    parser.add_argument('--coeff_marginal_regularization_min', type=float, default=6.0, help='min Coefficient for marginal regularization')
    parser.add_argument('--coeff_main_loss_term_min', type=float, default=1.0, help='min Coefficient for main loss term')
    parser.add_argument('--num_time_samples', type=int, default=40, help='Number of time samples')
    parser.add_argument('--num_time_samples_evaluation', type=int, default=40, help='Number of time samples')
    parser.add_argument('--seed', type=int, default=5678, help='Random seed')
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--temp', type=float) # EFF: add burn in as a parameter else tune

    args = parser.parse_args()
    LATTICE_SIZE_ISING = 32

    # Setup keys
    key = jr.PRNGKey(5678)
    model_key, loader_key, loss_key, test_key, evaluation_key, key_validation = jr.split(key, 6)

    OUTPUT_FILE_NAME = f"tuning{vars(args)}.json"
    OUTPUT_FILE_PTH = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)
    PLACEHOLDER_ISING_MEAN = jnp.zeros(LATTICE_SIZE_ISING)
    PLACEHOLDER_ISING_STD = jnp.ones(LATTICE_SIZE_ISING)
    # Constants calculation
    OLD_BATCH_SIZE = 500


    INTEGRATED_TIME = max(get_help_finding_int_time(args.temp, LATTICE_SIZE_ISING, 1.0, 1.0, n=100), LATTICE_SIZE_ISING)
    COEFF_FOR_BURN_IN=2
    BURN_IN = COEFF_FOR_BURN_IN*INTEGRATED_TIME

    #TAINTED================
    # Generate a dataset of size 19000
    NUM_TRAIN_SAMPLES = 100
    NUM_SAMPLES_TEST = 100
    NUM_SAMPLES_VALIDATION = 100
    NUM_CHAINS = 100
    assert NUM_TRAIN_SAMPLES % NUM_CHAINS == 0 and NUM_SAMPLES_VALIDATION % NUM_CHAINS == 0 and NUM_SAMPLES_TEST % NUM_CHAINS == 0
    dataset_key, test_key_new, loader_key = jr.split(loader_key, 3)

    full_dataset = sample_from_continuous_relaxation_1D(dataset_key, NUM_TRAIN_SAMPLES, LATTICE_SIZE_ISING, args.temp, INTEGRATED_TIME, BURN_IN, NUM_CHAINS)

    # Standardize the dataset
    full_dataset, dataset_mean, dataset_std = create_standardized_dataset(full_dataset)

    # Instantiate the regular DataLoader
    dataloader = DataLoader(full_dataset, NNRGIsingConfig.BATCH_SIZE, loader_key)

    # Generate test dataset
    test_dataset = sample_from_continuous_relaxation_1D(test_key_new, NUM_SAMPLES_TEST, LATTICE_SIZE_ISING, args.temp, INTEGRATED_TIME, BURN_IN, NUM_CHAINS)
    test_dataset = (test_dataset - dataset_mean) / dataset_std

    validation_dataset = sample_from_continuous_relaxation_1D(key_validation, NUM_SAMPLES_VALIDATION, LATTICE_SIZE_ISING, args.temp, INTEGRATED_TIME, BURN_IN, NUM_CHAINS )
    validation_dataset = (validation_dataset - dataset_mean) / dataset_std
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

    def make_and_save_visualizations_of_best_models(frontier, key_frontier):
        key_frontier_visualizations = jr.split(key_frontier, len(frontier))
        NUM_SAMPLES_BASIC_EVAL = 500
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
                        desc=f"hypersweep{tuple(parameters.items())}",
                        num_time_samples = args.num_time_samples,
                        num_time_samples_test=args.num_time_samples_evaluation,
                        )
            nrg_model = load_model(siren_model_dir + name_of_model, WrapperForNNRG)
            configs_sampled_from_model = get_discrete_samples_from_model(nrg_model, key_discrete_model, LATTICE_SIZE_ISING, NUM_SAMPLES_BASIC_EVAL)
            configs_from_test_dataset = get_discrete_samples(test_dataset[:NUM_SAMPLES_BASIC_EVAL], key_discrete_test)
            fig, stats = compare_model_vs_validation(configs_sampled_from_model, configs_from_test_dataset, n_show=40)
            fname = "OutputVis" + get_model_file_identifier(lr= parameters[LR_PARAM_NAME],
                        ke_schedule=KESchedule(parameters[PENALTY_COEFF_NAME], parameters[PARAM_NAME_STEPS_TIL_0]),
                        coeff_marginal_regularization=parameters[PARAM_NAME_MARGINAL_REGULARIZATION],
                        coeff_main_loss_term=parameters[PARAM_NAME_MAIN_TERM],
                        steps=args.steps,
                        desc=f"hypersweep{tuple(parameters.items())}",
                        num_time_samples = args.num_time_samples,
                        num_time_samples_test=args.num_time_samples_evaluation) + ".pdf"
            fig.savefig(os.path.join(OUTPUT_DIR, fname))



    # Configure and experiment with the desired parameters
    STEPS_IN_EPOCH = int(jnp.ceil(dataloader.array.shape[0]/dataloader.batch_size))
    client.configure_experiment(parameters=[
        RangeParameterConfig(
            name="penaltyCoeff",
            bounds=(1e-3, 2),
            parameter_type="float",
            scaling="log"
        ),
        RangeParameterConfig(
            name=PARAM_NAME_MARGINAL_REGULARIZATION,
            bounds=(6, 10),
            parameter_type="int",
        ),
        RangeParameterConfig(
            name="mainCoeff",
            bounds=(0.5, 2),
            parameter_type="float",
            scaling="log"
        ),
        RangeParameterConfig(
        name=PARAM_NAME_STEPS_TIL_0,
        bounds=(3*STEPS_IN_EPOCH, int(jnp.ceil(PATIENCE_NUM_EPOCHS*STEPS_IN_EPOCH/3))), # from 3 epochs to half
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
    outcome_constraints=[f"{NLL_METRIC_NAME} <= 300", f"{MMD_METRIC_NAME} <= 0.04", f"{KE_PENALTY_NAME} <= 0.2"],
)




    for _ in range(3):
        evaluation_key, key_nll, key_sample_quality, key_ot_penalty = jr.split(evaluation_key, 4)
        trials = client.get_next_trials(max_trials=1)
        per_trial_loss_msgs = []
        for trial_index, parameters in trials.items():

            # Model initialization
            nrg_model = WrapperForNNRG(depth=int(jnp.ceil(jnp.log2(LATTICE_SIZE_ISING))), key=model_key)


    
            nrg_model, (_, loss_msgs, _) = train_nnrg(
                nrg_model,
                dataloader,
                loss_key,
                lr=parameters[LR_PARAM_NAME],
                ke_penalty_coeff=parameters[PENALTY_COEFF_NAME],
                coeff_marginal_regularization=parameters[PARAM_NAME_MARGINAL_REGULARIZATION],
                coeff_main_loss_term=parameters[PARAM_NAME_MAIN_TERM],
                num_time_samples=args.num_time_samples,
                dataset_test = validation_dataset,
                num_time_samples_test= args.num_time_samples_evaluation,
                desc=str(tuple(parameters.items())) + str(args) + f"lsize{LATTICE_SIZE_ISING}",
                steps=args.steps,
                ke_schedule=KESchedule(parameters[PENALTY_COEFF_NAME], parameters[PARAM_NAME_STEPS_TIL_0]) 
            )
            per_trial_loss_msgs.append(loss_msgs)
            inference_info = ModelInferenceInfo(nrg_model, PLACEHOLDER_ISING_MEAN, PLACEHOLDER_ISING_STD)
            sample_quality = evaluate_sample_quality_nnrg(inference_info, test_dataset, key_sample_quality, LATTICE_SIZE_ISING)
            penalties = penalties_on_test_data(nrg_model, test_dataset, key_ot_penalty, args.num_time_samples_evaluation)
            raw_data = {NLL_METRIC_NAME: penalties["nll"], MMD_METRIC_NAME: sample_quality, KE_PENALTY_NAME: penalties["ke"]}

            client.complete_trial(trial_index=trial_index, raw_data=raw_data)

    frontier = client.get_pareto_frontier()
    make_and_save_visualizations_of_best_models(frontier, evaluation_key)
    with open(OUTPUT_FILE_PTH, "w") as file:
      json.dump({"frontier": make_json_serializable(frontier), "loss_msgs": per_trial_loss_msgs}, file)



if __name__ == '__main__':
    # To run in Colab without crashing on sys.argv:
    
    if 'ipykernel' in sys.modules:
        sys.argv = ['']
    main()
