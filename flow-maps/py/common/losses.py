"""
Nicholas M. Boffi
10/5/25

Loss functions for learning.
"""

import functools
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
from ml_collections import config_dict

from . import flow_map as flow_map
from . import interpolant as interpolant
from . import loss_args

Parameters = Dict[str, Dict]


def mean_reduce(func):
    """
    A decorator that computes the mean of the output of the decorated function.
    Designed to be used on functions that are already batch-processed (e.g., with jax.vmap).
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        batched_outputs = func(*args, **kwargs)
        return jnp.mean(batched_outputs)

    return wrapper


def diagonal_term(
    params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
) -> float:
    """Compute the diagonal (interpolant) term of the loss."""

    # compute interpolant and the target
    It = interp.calc_It(t, x0, x1)
    It_dot = interp.calc_It_dot(t, x0, x1)

    # compute the weighted loss
    bt = X.apply(params, t, It, label, train=True, method="calc_b", rngs=rng)
    velocity_loss = jnp.sum((bt - It_dot) ** 2)

    # Diagonal uses s=t
    weight_tt = X.apply(params, t, t, method="calc_weight")
    return jnp.exp(-weight_tt) * velocity_loss + weight_tt


def psd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    u: float,
    h: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    psd_type: str,
    stopgrad_type: str,
) -> float:
    """Compute the PSD (Progressive Self-Distillation) term of the loss."""
    Is = interp.calc_It(s, x0, x1)

    # compute the full jump
    X_st, phi_st = X.apply(
        params, s, t, Is, label, train=False, rngs=rng, return_X_and_phi=True
    )

    # break it down into two jumps
    if stopgrad_type == "convex":
        X_su, phi_su = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                s,
                u,
                Is,
                label,
                train=False,
                rngs=rng,
                return_X_and_phi=True,
            )
        )

        X_ut, phi_ut = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                u,
                t,
                X_su,
                label,
                train=False,
                rngs=rng,
                return_X_and_phi=True,
            )
        )
    elif stopgrad_type == "none":
        X_su, phi_su = X.apply(
            params,
            s,
            u,
            Is,
            label,
            train=False,
            rngs=rng,
            return_X_and_phi=True,
        )

        X_ut, phi_ut = X.apply(
            params,
            u,
            t,
            X_su,
            label,
            train=False,
            rngs=rng,
            return_X_and_phi=True,
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    if psd_type == "uniform":
        student = phi_st
        teacher = (1 - h) * phi_su + h * phi_ut
    elif psd_type == "midpoint":
        student = phi_st
        teacher = 0.5 * (phi_su + phi_ut)
    else:
        raise ValueError(f"Invalid psd_type: {psd_type}")

    psd_loss = jnp.sum((student - teacher) ** 2)

    weight_st = X.apply(params, s, t, method="calc_weight")
    return jnp.exp(-weight_st) * psd_loss + weight_st


def lsd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    stopgrad_type: str,
) -> float:
    """Compute the LSD term of the loss."""
    Is = interp.calc_It(s, x0, x1)

    # Compute the distillation loss
    Xst_Is, dt_Xst = X.apply(
        params, s, t, Is, label, train=False, method="partial_t", rngs=rng
    )

    if stopgrad_type == "convex":
        Xst_Is = jax.lax.stop_gradient(Xst_Is)
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                t,
                Xst_Is,
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )
    elif stopgrad_type == "none":
        b_eval = X.apply(
            params,
            t,
            Xst_Is,
            label,
            train=False,
            method="calc_b",
            rngs=rng,
        )
    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    weight_st = X.apply(params, s, t, method="calc_weight")
    error = b_eval - dt_Xst
    lsd_loss = jnp.sum(error**2)
    return jnp.exp(-weight_st) * lsd_loss + weight_st


def esd_term(
    params: Parameters,
    teacher_params: Parameters,
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    label: jnp.ndarray,
    s: float,
    t: float,
    rng: jnp.ndarray,
    *,
    interp: interpolant.Interpolant,
    X: flow_map.FlowMap,
    stopgrad_type: str,
) -> float:
    """Compute the ESD term of the loss."""
    Is = interp.calc_It(s, x0, x1)

    # compute the derivative with respect to the first time
    _, ds_Xst = X.apply(
        params, s, t, Is, label, train=False, method="partial_s", rngs=rng
    )

    # stopgrad everything to avoid backpropagating through the UNet spatial Jacobian
    if stopgrad_type == "full":
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                s,
                Is,
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )

        # compute the advective term
        _, grad_Xst_b = jax.lax.stop_gradient(
            jax.jvp(
                lambda x: X.apply(
                    teacher_params, s, t, x, label, train=False, rngs=rng
                ),
                primals=(Is,),
                tangents=(b_eval,),
            )
        )

    # stopgrad the b, so it's like EMD
    elif stopgrad_type == "convex":
        b_eval = jax.lax.stop_gradient(
            X.apply(
                teacher_params,
                s,
                Is,
                label,
                train=False,
                method="calc_b",
                rngs=rng,
            )
        )

        # compute the advective term
        _, grad_Xst_b = jax.jvp(
            lambda x: X.apply(params, s, t, x, label, train=False, rngs=rng),
            primals=(Is,),
            tangents=(b_eval,),
        )

    # pure residual minimization -- no stopgrad
    elif stopgrad_type == "none":
        b_eval = X.apply(
            params,
            s,
            Is,
            label,
            train=False,
            method="calc_b",
            rngs=rng,
        )

        # compute the advective term
        _, grad_Xst_b = jax.jvp(
            lambda x: X.apply(params, s, t, x, label, train=False, rngs=rng),
            primals=(Is,),
            tangents=(b_eval,),
        )

    else:
        raise ValueError(f"Invalid stopgrad_type: {stopgrad_type}")

    esd_loss = jnp.sum((ds_Xst + grad_Xst_b) ** 2)
    weight_st = X.apply(params, s, t, method="calc_weight")
    return jnp.exp(-weight_st) * esd_loss + weight_st


def setup_loss(
    cfg: config_dict.ConfigDict, net: flow_map.FlowMap, interp: interpolant.Interpolant
) -> Callable:
    """Setup the loss function."""

    print(f"Setting up loss: {cfg.training.loss_type}")
    print(f"Stopgrad type: {cfg.training.stopgrad_type}")

    # Pure diagonal loss
    @mean_reduce
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0))
    def diagonal_only_loss(params, x0, x1, label, t, rng):
        return diagonal_term(
            params,
            x0,
            x1,
            label,
            t,
            rng,
            interp=interp,
            X=net,
        )

    # Pure off-diagonal loss
    @mean_reduce
    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0))
    def offdiagonal_only_loss(
        params, teacher_params, x0, x1, label, s, t, u, h, dropout_keys
    ):
        rng = {"dropout": dropout_keys}

        if cfg.training.loss_type == "psd":
            return psd_term(
                params,
                teacher_params,
                x0,
                x1,
                label,
                s,
                t,
                u,
                h,
                rng,
                interp=interp,
                X=net,
                psd_type=cfg.training.psd_type,
                stopgrad_type=cfg.training.stopgrad_type,
            )
        elif cfg.training.loss_type == "lsd":
            return lsd_term(
                params,
                teacher_params,
                x0,
                x1,
                label,
                s,
                t,
                rng,
                interp=interp,
                X=net,
                stopgrad_type=cfg.training.stopgrad_type,
            )
        elif cfg.training.loss_type == "esd":
            return esd_term(
                params,
                teacher_params,
                x0,
                x1,
                label,
                s,
                t,
                rng,
                interp=interp,
                X=net,
                stopgrad_type=cfg.training.stopgrad_type,
            )
        else:
            raise ValueError(f"Unknown loss_type: {cfg.training.loss_type}")

    def loss(params, teacher_params, x0, x1, label, s, t, u, h, dropout_keys):
        """Split batch into diagonal and off-diagonal portions."""
        total_bs = x0.shape[0]
        diag_bs, offdiag_bs = loss_args._get_diag_offdiag_bs(cfg, total_bs)

        total_loss = 0.0

        # Compute diagonal loss on first portion
        if diag_bs > 0:
            label_diag = None if label is None else label[:diag_bs]
            diag_loss = diagonal_only_loss(
                params,
                x0[:diag_bs],
                x1[:diag_bs],
                label_diag,
                t[:diag_bs],
                dropout_keys[:diag_bs],
            )
            total_loss += diag_loss * diag_bs

        # Compute off-diagonal loss on second portion
        if offdiag_bs > 0:
            label_offdiag = None if label is None else label[diag_bs:]
            u_offdiag = None if u is None else u[diag_bs:]
            h_offdiag = None if h is None else h[diag_bs:]

            offdiag_loss = offdiagonal_only_loss(
                params,
                teacher_params,
                x0[diag_bs:],
                x1[diag_bs:],
                label_offdiag,
                s[diag_bs:],
                t[diag_bs:],
                u_offdiag,
                h_offdiag,
                dropout_keys[diag_bs:],
            )
            total_loss += offdiag_loss * offdiag_bs

        # Normalize by total batch size
        return total_loss / total_bs

    return loss
