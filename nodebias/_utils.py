import jax

from ._config import *

import jax.numpy as jnp

import numpy as np
np.set_printoptions(suppress=True)

import equinox as eqx
import diffrax

import matplotlib.pyplot as plt

import optax
from functools import partial

import os
import time
# import cProfile







def seconds_to_hours(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return hours, minutes, seconds

## Simply returns a suitable key for all jax operations
def get_new_key(key=None, num=1):
    if key is None:
        print("WARNING: No key provided, using time as seed")
        key = jax.random.PRNGKey(time.time_ns())

    elif isinstance(key, int):
        key = jax.random.PRNGKey(key)

    keys = jax.random.split(key, num=num)

    return keys if num > 1 else keys[0]

def generate_new_keys(key=None, num=1):
    if key is None:
        print("WARNING: No key provided, using time as seed")
        key = jax.random.PRNGKey(time.time_ns())

    elif isinstance(key, int):
        key = jax.random.PRNGKey(key)

    return jax.random.split(key, num=num)

## Wrapper function for matplotlib and seaborn
def sbplot(*args, ax=None, figsize=(6,3.5), x_label=None, y_label=None, title=None, x_scale='linear', y_scale='linear', xlim=None, ylim=None, **kwargs):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(context='notebook', style='ticks',
            font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})
    plt.style.use("dark_background")

    if ax==None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
    # sns.despine(ax=ax)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.plot(*args, **kwargs)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    if "label" in kwargs.keys():
        ax.legend()
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    plt.tight_layout()
    return ax

## Alias for sbplot
def plot(*args, ax=None, figsize=(6,3.5), x_label=None, y_label=None, title=None, x_scale='linear', y_scale='linear', xlim=None, ylim=None, **kwargs):
  return sbplot(*args, ax=ax, figsize=figsize, x_label=x_label, y_label=y_scale, title=title, x_scale=x_scale, y_scale=y_scale, xlim=xlim, ylim=ylim, **kwargs)


def pvplot(x, y, show=True, xlabel=None, ylabel=None, title=None, ax=None, **kwargs):
    import pyvista as pv
    # pv.start_xvfb()           ## TODO Only do this on LINUX

    if ax is None:
        ax = pv.Chart2D()

    _ = ax.line(x, y, **kwargs)

    if xlabel is not None:
        ax.x_label = xlabel

    if ylabel is not None:
        ax.y_label = ylabel

    if title is not None:
        ax.title = title

    if show == True:
        ax.show()

    return ax


def flatten_pytree(pytree):
    """ Flatten the leaves of a pytree into a single array. Return the array, the shapes of the leaves and the tree_def. """

    leaves, tree_def = jax.tree_util.tree_flatten(pytree)
    flat = jnp.concatenate([x.flatten() for x in leaves])
    shapes = [x.shape for x in leaves]
    return flat, shapes, tree_def

def unflatten_pytree(flat, shapes, tree_def):
    """ Reconstructs a pytree given its leaves flattened, their shapes, and the treedef. """

    leaves_prod = [0]+[np.prod(x) for x in shapes]

    lpcum = np.cumsum(leaves_prod)
    leaves = [flat[lpcum[i-1]:lpcum[i]].reshape(shapes[i-1]) for i in range(1, len(lpcum))]

    return jax.tree_util.tree_unflatten(tree_def, leaves)
