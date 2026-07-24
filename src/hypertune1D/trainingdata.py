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
import hashlib
import warnings

import jax

import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import norm

import matplotlib.pyplot as plt

import equinox as eqx  # https://github.com/patrick-kidger/equinox


import scipy.stats as stats


import matplotlib.pyplot as plt  # Used for creating static, interactive, and animated visualizations in Python.
from sklearn.datasets import make_circles, make_moons

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

from numpyro.diagnostics import effective_sample_size

J_test = 1.

def energy( s, coupling) :
    '''!Returns the energy divided by the number of sites of the configuration `s` , assuming that the configuration is from a
    1D Ising model with periodic boundary conditions and that is subject to no external magnetic field'''
    # this is the energy for each site
    E = -coupling * ( s * jnp.roll( s, 1 ) )
    # and this is the avg energy per site
    return jnp.sum( E ) / s.size

#Creates an LxL lattice of random integer spins with probability (p) to be +1 (and 1-p to be -1)
def randomLattice(key, L, p ) :

    return ( jr.uniform(key, L ) < p ) * 2 - 1

def magnetization(configuration):
    '''Returns the microscopic-state magnetization of the 1D Ising model `configuration` (assuming that it is not subject to
    an external magnetic field) which is defined as the sum of the configuraton's spins'''
    return jnp.sum( configuration )

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

# Vectorized Metropolis across parallel chains
metropolis = jax.vmap(metropolis_single_chain, in_axes=(0, 0, None, None))


def get_help_finding_int_time(key, T, L, kB, J, n=1000, p=0.5):
    seed = 170
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

def sample_from_low_temp_ising(key, dataset_size, lattice_size):
    dataset = jnp.zeros((dataset_size, lattice_size))
    selectors = jr.bernoulli(key, shape=(dataset_size,))
    return selectors[:, None].astype(jnp.float32) * jnp.ones((dataset_size, lattice_size))


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


def zero_temp_sample_from_continuous_relaxation_1D(key, dataset_size, lattice_size):
  key_contin, key_discrete = jr.split(key)
  key_contin = jr.split(key_contin, dataset_size)
  samples = sample_from_low_temp_ising(key_discrete, dataset_size, lattice_size)
  K, alpha = get_K_alpha(lattice_size, 0)
  dataset_continuous = jax.vmap(sample_continuous_from_discrete, (0, 0, None, None, None))(key_contin, samples, K, alpha, lattice_size)
  return dataset_continuous

def sample_from_continuous_relaxation_1D(key, dataset_size, lattice_size, temp, integrated_time, burn_in, num_chains):
    if temp == 0:
        return zero_temp_sample_from_continuous_relaxation_1D(key, dataset_size, lattice_size)
    
    discrete_key, key_continuous = jr.split(key, 2)
    dataset_discrete = sample_discrete_configurations(discrete_key, dataset_size, lattice_size, temp, integrated_time, burn_in, num_chains)
    keys_continuous = jr.split(key_continuous, dataset_size)
    K, alpha = get_K_alpha(lattice_size, temp)
    dataset_continuous = jax.vmap(sample_continuous_from_discrete, (0, 0, None, None, None))(keys_continuous, dataset_discrete, K, alpha, lattice_size)
    return dataset_continuous


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





def main():
    parser = argparse.ArgumentParser(description='Generate data.')
    parser.add_argument('--lattice_size', required=True,type=int)
    parser.add_argument('--num_train_samples', required=True,type=int)
    parser.add_argument('--num_test_samples',required=True, type=int)
    parser.add_argument('--num_val_samples', required=True,type=int)
    parser.add_argument('--temp', required=True,type=float) #
    parser.add_argument('--out', required=True,help='should be a directory with longterm storage (so not local scratch)')

    args = parser.parse_args()
  
    seed = 5678
    key = jr.PRNGKey(seed)

    test_key, dataset_key = jr.split(key)

    OUTPUT_DIR = args.out

    INTEGRATED_TIME = None if args.temp == 0 else max(get_help_finding_int_time(test_key, args.temp, args.lattice_size, 1.0, 1.0, n=1000), args.lattice_size)
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
    

    DATA_CACHE_FILE_NAME = "dataset_cache_" + hashlib.md5(
        f"data{args.lattice_size}_{args.temp}_{NUM_TRAIN_SAMPLES}_{NUM_SAMPLES_TEST}_{NUM_SAMPLES_VALIDATION}_{seed}".encode('utf-8')
    ).hexdigest() + ".npz"
    DATA_CACHE_PATH = os.path.join(OUTPUT_DIR, DATA_CACHE_FILE_NAME)

    if os.path.exists(DATA_CACHE_PATH):
        cached = np.load(DATA_CACHE_PATH)
        full_dataset = jnp.array(cached["full_dataset"])
        dataset_mean = jnp.array(cached["dataset_mean"])
        dataset_std = jnp.array(cached["dataset_std"])
        test_dataset = jnp.array(cached["test_dataset"])
        validation_dataset = jnp.array(cached["validation_dataset"])
    else:

        all_data = sample_from_continuous_relaxation_1D(dataset_key, NUM_TRAIN_SAMPLES + NUM_SAMPLES_TEST + NUM_SAMPLES_VALIDATION, args.lattice_size, args.temp, INTEGRATED_TIME, BURN_IN, NUM_CHAINS)

        full_dataset = all_data[:NUM_TRAIN_SAMPLES]
        test_dataset = all_data[NUM_TRAIN_SAMPLES:NUM_TRAIN_SAMPLES+NUM_SAMPLES_TEST]
        validation_dataset = all_data[NUM_TRAIN_SAMPLES+NUM_SAMPLES_TEST:NUM_TRAIN_SAMPLES+NUM_SAMPLES_TEST+NUM_SAMPLES_VALIDATION]
        
    
    def _stats(spins):
        magnetization = jnp.sum(spins, axis=1)  # per-sample magnetization
        nn_corr = jnp.mean(spins[:, :-1] * spins[:, 1:])  # <s_i s_{i+1}>, averaged
        return {
            "mean_magnetization": float(jnp.mean(magnetization)),
            "std_magnetization": float(jnp.std(magnetization)),
            "mean_nn_correlation": float(nn_corr),
        }
    print(_stats(full_dataset))
    

if __name__ == '__main__':
    # To run in Colab without crashing on sys.argv:
    
    if 'ipykernel' in sys.modules:
        # rendering the following as a sys.arvlist: --batch_size=50 --steps=100 --temp=199 --num_trials=3 --num_train_samples=100 --num_test_samples=100 --num_val_samples=100 --lattice_size=32
        sys.argv = ['', '--temp=199', '--num_train_samples=100', '--num_test_samples=100', '--num_val_samples=100', '--lattice_size=32', '--out=/',]
    main()