"""
Nicholas M. Boffi
10/5/25

Helper routines for neural network definitions.
"""

from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections import config_dict

from . import edm2_net as edm2_net


class MLP(nn.Module):
    """Simple MLP network with square weight pattern."""

    n_hidden: int
    n_neurons: int
    output_dim: int
    act: Callable
    use_residual: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.n_neurons)(x)
        x = self.act(x)

        for _ in range(self.n_hidden):
            if self.use_residual:
                x = x + nn.Dense(self.n_neurons)(x)
            else:
                x = nn.Dense(self.n_neurons)(x)
            x = self.act(x)

        x = nn.Dense(self.output_dim)(x)
        return x


class FlowMapMLP(nn.Module):
    """Simple MLP network with square weight pattern, for flow map representation."""

    config: config_dict.ConfigDict

    def setup(self):
        self.phi_mlp = MLP(
            self.config.n_hidden,
            self.config.n_neurons,
            self.config.output_dim,
            get_act(self.config),
            self.config.use_residual,
        )

        self.weight_mlp = MLP(
            n_hidden=1,
            n_neurons=self.config.n_neurons,
            output_dim=1,
            act=jax.nn.gelu,
            use_residual=False,
        )

    def calc_weight(self, s: float, t: float) -> float:
        st = jnp.array([s, t])
        # return self.weight_mlp(st)
        return 1.0

    def calc_phi(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        init_weights: bool = False,
    ) -> jnp.ndarray:
        del label
        del train
        del init_weights  # MLP doesn't have dual weights to initialize
        st = jnp.array([s, t])
        inp = jnp.concatenate((st, x / self.config.rescale))
        phi_st = self.config.rescale * self.phi_mlp(inp)

        if calc_weight:
            weight = self.calc_weight(s, t)
            return phi_st, weight
        else:
            return phi_st

    def calc_b(
        self,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
    ) -> jnp.ndarray:
        return self.calc_phi(t, t, x, label, train, calc_weight)

    def __call__(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight=False,
        return_X_and_phi: bool = False,
        init_weights: bool = False,
    ) -> jnp.ndarray:
        del label
        phi_st = self.calc_phi(
            s,
            t,
            x,
            label=None,
            train=train,
            calc_weight=calc_weight,
            init_weights=init_weights,
        )
        if calc_weight:
            phi_st, weight = phi_st

        X_st = x + (t - s) * phi_st

        if calc_weight:
            return X_st, weight
        elif return_X_and_phi:
            return X_st, phi_st
        else:
            return X_st


class EDM2FlowMap(nn.Module):
    """UNet architecture based on EDM2.
    Note: assumes that there is no batch dimension, to interface with the rest of the code.
    Adds a padded batch dimension to handle this.
    """

    config: config_dict.ConfigDict

    def setup(self):
        self.one_hot_dim = (
            self.config.label_dim + 1 if self.config.use_cfg else self.config.label_dim
        )
        self.net = edm2_net.PrecondFlowMap(
            img_resolution=self.config.img_resolution,
            img_channels=self.config.img_channels,
            label_dim=self.one_hot_dim,
            sigma_data=self.config.rescale,
            logvar_channels=self.config.logvar_channels,
            use_bfloat16=self.config.use_bfloat16,
            use_weight=self.config.use_weight,
            unet_kwargs=self.config.unet_kwargs,
        )

    def process_inputs(self, s: float, t: float, x: jnp.ndarray, label: float = None):
        # add batch dimensions
        s = jnp.asarray(s, dtype=jnp.float32)
        t = jnp.asarray(t, dtype=jnp.float32)
        x = x.reshape((1, *x.shape))

        # one-hot encode
        if label != None:
            label = jax.nn.one_hot(label, num_classes=self.one_hot_dim).reshape((1, -1))

        return s, t, x, label

    def calc_weight(self, s: float, t: float) -> jnp.ndarray:
        # add batch dimension
        s = jnp.asarray(s, dtype=jnp.float32)
        t = jnp.asarray(t, dtype=jnp.float32)
        return self.net.calc_weight(s, t)

    def calc_phi(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        init_weights: bool = False,
    ) -> jnp.ndarray:
        s, t, x, label = self.process_inputs(s, t, x, label)
        rslt = self.net.calc_phi(s, t, x, label, train, calc_weight, init_weights)
        if calc_weight:
            Xst, logvar = rslt
            return Xst[0], logvar[0]
        else:
            return rslt[0]

    def calc_b(
        self,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
    ) -> jnp.ndarray:
        _, t, x, label = self.process_inputs(t, t, x, label)
        rslt = self.net.calc_b(t, x, label, train, calc_weight)
        if calc_weight:
            bt, logvar = rslt
            return bt[0], logvar[0]
        else:
            return rslt[0]

    def __call__(
        self,
        s: float,
        t: float,
        x: jnp.ndarray,
        label: float = None,
        train: bool = True,
        calc_weight: bool = False,
        return_X_and_phi: bool = False,
        init_weights: bool = False,
    ):
        s, t, x, label = self.process_inputs(s, t, x, label)
        rslt = self.net(
            s, t, x, label, train, calc_weight, return_X_and_phi, init_weights
        )

        if calc_weight:
            Xst, logvar = rslt
            return Xst[0], logvar[0]
        elif return_X_and_phi:
            Xst, phi_st = rslt
            return Xst[0], phi_st[0]
        else:
            Xst = rslt
            return Xst[0]


def get_act(
    config: config_dict.ConfigDict,
) -> Callable:
    """Get the activation function for the network.

    Args:
        config: Configuration dictionary.
    """
    if config.act == "gelu":
        return jax.nn.gelu
    elif config.act == "swish" or config.act == "silu":
        return jax.nn.silu
    else:
        raise ValueError(f"Activation function {config.activation} not recognized.")


def setup_network(
    network_config: config_dict.ConfigDict,
) -> nn.Module:
    """Setup the neural network for the system.

    Args:
        config: Configuration dictionary.
    """
    if "mlp" in network_config.network_type:
        return FlowMapMLP(config=network_config)
    elif network_config.network_type == "edm2":
        return EDM2FlowMap(config=network_config)
    else:
        raise ValueError(f"Network type {network_config.network_type} not recognized.")
