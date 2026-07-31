"""
Nicholas M. Boffi
10/5/25

Utilities for storing training state.
"""

from copy import deepcopy
from typing import Any, Callable, Dict, NamedTuple, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
from flax import struct
from flax.serialization import from_bytes
from flax.training import train_state
from ml_collections import config_dict

from . import flow_map, interpolant
from . import dist_utils


class EMATrainState(train_state.TrainState):
    """Train state including EMA parameters."""

    ema_params: Dict[float, Any] = struct.field(default_factory=dict)


class StaticArgs(NamedTuple):
    net: nn.Module
    schedule: optax.Schedule
    loss: Callable
    get_loss_fn_args: Callable
    train_step: Callable
    update_ema_params: Callable
    ds: tf.data.Dataset
    interp: interpolant.Interpolant
    sample_rho0: Callable
    inception_fn: Callable = None  # For FID computation


def load_checkpoint(
    cfg: config_dict.ConfigDict,
    train_state: EMATrainState,
) -> EMATrainState:
    """Load a training checkpoint."""
    with open(cfg.network.load_path, "rb") as f:
        raw_bytes = f.read()
        train_state = from_bytes(train_state, raw_bytes)

    return train_state


def setup_schedule(
    cfg: config_dict.ConfigDict,
) -> optax.Schedule:
    """Set up the learning rate schedule."""
    if cfg.optimization.schedule_type == "cosine":
        return optax.cosine_decay_schedule(
            init_value=cfg.optimization.learning_rate,
            decay_steps=cfg.optimization.decay_steps,
            alpha=0.0,
        )
    elif cfg.optimization.schedule_type == "sqrt":
        return lambda step: cfg.optimization.learning_rate / jnp.sqrt(
            jnp.maximum(step / cfg.optimization.decay_steps, 1.0)
        )
    elif cfg.optimization.schedule_type == "constant":
        return lambda step: cfg.optimization.learning_rate
    else:
        raise ValueError(f"Unknown schedule type: {cfg.schedule_type}")


def setup_optimizer(cfg: config_dict.ConfigDict):
    """Set up the optimizer."""
    schedule = setup_schedule(cfg)

    # optimizer mask for positional embeddings (which do not have a constants key)
    def mask_fn(variables):
        masks = {
            "params": jax.tree_util.tree_map(lambda _: True, variables["params"]),
        }
        if "constants" in variables:  # network has Fourier tables
            masks["constants"] = jax.tree_util.tree_map(
                lambda _: False, variables["constants"]
            )
        return masks

    # define optimizer
    tx = optax.masked(
        optax.chain(
            optax.clip_by_global_norm(cfg.optimization.clip),
            optax.radam(
                learning_rate=schedule,
                b1=getattr(cfg.optimization, "b1", 0.9),
                b2=getattr(cfg.optimization, "b2", 0.999),
                eps=getattr(cfg.optimization, "eps", 1e-8),
            ),
        ),
        mask_fn,
    )

    return tx, schedule


def setup_training_state(
    cfg: config_dict.ConfigDict,
    ex_input: jnp.ndarray,
    prng_key: jnp.ndarray,
) -> Tuple[EMATrainState, flow_map.FlowMap, optax.Schedule, jnp.ndarray]:
    """Load flax training state."""

    # Initialize flow map network
    net, params, prng_key = flow_map.initialize_flow_map(
        cfg.network, ex_input, prng_key
    )
    ema_params = {ema_fac: deepcopy(params) for ema_fac in cfg.training.ema_facs}

    # define training state
    tx, schedule = setup_optimizer(cfg)
    train_state = EMATrainState.create(
        apply_fn=net.apply,
        params=params,
        ema_params=ema_params,
        tx=tx,
    )

    # load training state from checkpoint, if desired
    if cfg.network.load_path != "":
        print("Loading full training state checkpoint.")
        train_state = load_checkpoint(cfg, train_state)
        print("Loaded training state checkpoint.")

        if cfg.network.reset_optimizer:
            print("Resetting optimizer state.")
            train_state = train_state.replace(
                opt_state=tx.init(train_state.params),
                step=0,
            )

    return train_state, net, schedule, prng_key
