"""
Nicholas M. Boffi
10/5/25

Update functions for learning.
"""

import functools
from typing import Callable, Dict, Tuple

import jax
import ml_collections.config_dict as config_dict
from jax import value_and_grad

from . import state_utils, edm2_net

Parameters = Dict[str, Dict]


def setup_train_step(cfg: config_dict.ConfigDict) -> Callable:
    """Setup the training step function for single or multi-device training."""

    if cfg.training.ndevices > 1:
        decorator = lambda f: jax.pmap(
            f,
            in_axes=(0, None, 0),
            static_broadcasted_argnums=(1,),
            axis_name="data",
        )
    else:
        decorator = lambda f: functools.partial(jax.jit, static_argnums=(1,))(f)

    @decorator
    def train_step(
        state: state_utils.EMATrainState,
        loss_func: Callable[[Parameters], float],
        loss_func_args=tuple(),
    ) -> Tuple[state_utils.EMATrainState, float, Parameters]:
        """Single training step for the neural network.

        Args:
            state: Training state.
            loss_func: Loss function for the parameters.
            loss_func_args: Argument other than the parameters for the loss function.
        """
        loss_value, grads = value_and_grad(loss_func)(state.params, *loss_func_args)

        if cfg.training.ndevices > 1:
            loss_value = jax.lax.pmean(loss_value, axis_name="data")
            grads = jax.lax.pmean(grads, axis_name="data")

        state = state.apply_gradients(grads=grads)

        # project for the edm2 network
        state = state.replace(params=edm2_net.safe_project_to_sphere(cfg, state.params))

        return state, loss_value, grads

    return train_step


def setup_ema_update(
    cfg: config_dict.ConfigDict,
) -> Callable:
    """Setup the function for updating the EMA parameters on single or multiple devices."""

    decorator = jax.jit if cfg.training.ndevices == 1 else jax.pmap

    @decorator
    def update_ema_params(
        state: state_utils.EMATrainState,
    ) -> state_utils.EMATrainState:
        """Update EMA parameters."""
        new_ema_params = {}
        for ema_fac, ema_params in state.ema_params.items():
            new_ema_params[ema_fac] = jax.tree_util.tree_map(
                lambda param, ema_param: ema_fac * ema_param + (1 - ema_fac) * param,
                state.params,
                ema_params,
            )

        return state.replace(ema_params=new_ema_params)

    return update_ema_params
