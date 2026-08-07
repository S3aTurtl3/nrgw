import argparse
import jax
import jax.numpy as jnp
import jax.random as jr
# Ensure all your classes and functions (WrapperForNNRG, train_nnrg, etc.) are imported or defined above this in the real script.


import sys


from ax.api.client import Client
from ax.api.configs import ChoiceParameterConfig, RangeParameterConfig

from diffequsolvewrapper import differential_equation_solve, differential_equation_solve_with_saveat
import math
import os
import pathlib
import time
import hashlib
import wandb
import warnings


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



import math
from dataclasses import dataclass



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
        sol = differential_equation_solve(
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
        sol = differential_equation_solve(
            term, solver, self.t1, self.t0, -self.dt0, y, (eps, self.func)
        )
        (y,), (delta_log_likelihood,) = sol.ys
        return y, delta_log_likelihood

    # Runs forward-in-time to draw samples from the CNF.
    def sample(self, *, key):
        y = jr.normal(key, (self.data_size,))
        term = diffrax.ODETerm(self.func)
        solver = diffrax.Tsit5()
        sol = differential_equation_solve(term, solver, self.t0, self.t1, self.dt0, y) #--- how to go forward in time
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
            sol = differential_equation_solve(term, solver, self.t0, self.t1, self.dt0, initial_values_for_y_and_delta_logp, args_for_term)
            return sol.ys

        def backward_branch(_):
            sol =  differential_equation_solve(term, solver, self.t1, self.t0, -self.dt0, initial_values_for_y_and_delta_logp, args_for_term)
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
            sol = differential_equation_solve(term, solver, self.t0, self.t1, self.dt0, initial_values_for_y_and_delta_logp, args_for_term)
            return sol.ys

        def backward_branch(_):
            sol =  differential_equation_solve(term, solver, self.t1, self.t0, -self.dt0, initial_values_for_y_and_delta_logp, args_for_term)
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
          sol = differential_equation_solve_with_saveat(term, solver, self.t0, self.t1, self.dt0, initial_values_for_y_and_delta_logp, args_for_term, saveat=saveat)
          return sol

      def backward_branch(_):
          sol =  differential_equation_solve_with_saveat(term, solver, self.t1, self.t0, -self.dt0, initial_values_for_y_and_delta_logp, args_for_term, saveat=saveat)
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
        sol = differential_equation_solve_with_saveat(
            term, solver, self.t0, self.t1, self.dt0, y, saveat=saveat
        )
        out.append(sol.ys)
        y = sol.ys[-1]
        out = jnp.concatenate(out) #--- shape probs 6 by 2
        assert len(out) == 6  # number of points we saved at
        return out

