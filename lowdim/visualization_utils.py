import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Callable, Optional, Sequence, Tuple, Union

from sampling_utils import sample_target



@torch.no_grad()
def _hist_bins(data, bins: int = 40):
    """Bin spec for ``ax.hist`` that tolerates constant data.

    Matplotlib raises "Too many bins for data range" when every sample is
    identical. That happens for genuinely degenerate controls — e.g. FMRG-E on
    a linear reward, where grad r is constant so every particle receives the
    same displacement.
    """
    a = np.asarray(data, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return bins
    lo, hi = float(a.min()), float(a.max())
    if hi - lo <= 0.0:
        pad = max(abs(lo), 1.0) * 1e-6
        return np.linspace(lo - pad, hi + pad, 3)
    return bins


def reward_visualization(
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    extent: float = 1.5,
    grid: int = 200,
    title: Optional[str] = None,
    contour_title: Optional[str] = None,
    scatter_groups: Optional[Sequence[dict]] = None,
    probe_points: Optional[Sequence[Tuple[str, torch.Tensor]]] = None,
    contour_levels: Optional[Sequence[float]] = None,
    cmap: str = "YlOrRd",
    figsize: Tuple[float, float] = (13, 5.5),
    show: bool = True,
    device: Optional[torch.device] = None,
    return_data: bool = False,
):
    """
    Visualise a 2-D reward r(x) as filled contours (+ optional scatter) and a 3-D surface.

    Args:
        reward_fn:     maps (B, 2) positions to (B,) or (B, 1) rewards.
        extent:        half-width of the square plot domain [-extent, extent]^2.
        grid:          number of grid points per axis for contour/surface evaluation.
        title:         optional figure suptitle.
        contour_title: optional title for the 2-D panel.
        scatter_groups: optional list of dicts with keys
                        points (N, 2), label, color, alpha, s, zorder.
        probe_points:  optional (name, point) pairs printed after plotting;
                       point may be shape (2,) or (1, 2).
        contour_levels: line-contour levels; defaults to [0.1, 0.3, 0.5, 0.7, 0.9].
        cmap:          matplotlib colormap name.
        figsize:       figure size.
        show:          call plt.show() when True.
        device:        device for grid/probe evaluation (CPU if None).
        return_data:   if True, return (fig, XX, YY, R_grid); otherwise return None
                       (avoids Jupyter printing large arrays as cell output).

    Returns:
        (fig, XX, YY, R_grid) when return_data=True, else None.
    """
    import matplotlib.pyplot as plt

    xs = np.linspace(-extent, extent, grid)
    ys = np.linspace(-extent, extent, grid)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    grid_pts = torch.tensor(
        np.stack([XX.ravel(), YY.ravel()], axis=1), dtype=torch.float32, device=device
    )

    R = reward_fn(grid_pts)
    if R.ndim == 2 and R.shape[1] == 1:
        R = R.squeeze(-1)
    R_grid = R.detach().cpu().numpy().reshape(grid, grid)

    fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"wspace": 0.25})
    fig.delaxes(axes[1])
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")

    ax = axes[0]
    cf = ax.contourf(XX, YY, R_grid, levels=30, cmap=cmap, alpha=0.85)
    fig.colorbar(cf, ax=ax, shrink=0.85, label="$r(x)$")
    levels = contour_levels if contour_levels is not None else [0.1, 0.3, 0.5, 0.7, 0.9]
    ax.contour(XX, YY, R_grid, levels=levels, colors="k", linewidths=0.6, alpha=0.5)

    if scatter_groups:
        for i, grp in enumerate(scatter_groups):
            pts = grp["points"]
            if isinstance(pts, torch.Tensor):
                pts = pts.detach().cpu().numpy()
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=grp.get("s", 4),
                alpha=grp.get("alpha", 0.4),
                c=grp.get("color", f"C{i}"),
                label=grp.get("label"),
                zorder=grp.get("zorder", 3 + i),
            )
        ax.legend(loc="upper right", fontsize=8, markerscale=2)

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_title(contour_title or "Reward contours")

    ax3.plot_surface(XX, YY, R_grid, cmap=cmap, alpha=0.9, linewidth=0, antialiased=True)
    ax3.set_xlabel("$x_1$")
    ax3.set_ylabel("$x_2$")
    ax3.set_zlabel("$r(x)$")
    ax3.set_title("Reward surface")
    ax3.view_init(elev=35, azim=-60)

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    if show:
        plt.show()

    if probe_points:
        for name, pt in probe_points:
            pt = pt.unsqueeze(0) if pt.ndim == 1 else pt
            if device is not None:
                pt = pt.to(device)
            val = reward_fn(pt).detach().cpu()
            print(f"{name:<28}: {val.item() if val.numel() == 1 else val.numpy()}")

    if return_data:
        return fig, XX, YY, R_grid
    return None





# ── Visualise controlled vs uncontrolled trajectories and control effort ────────


def visualize_control_results(
    *,
    traj_ctrl: Union[torch.Tensor, np.ndarray],
    traj_unc: Union[torch.Tensor, np.ndarray],
    x0: Union[torch.Tensor, np.ndarray],
    xN_ctrl: Union[torch.Tensor, np.ndarray],
    xN_unc: Union[torch.Tensor, np.ndarray],
    controls: Optional[Union[torch.Tensor, np.ndarray]] = None,
    mu_history: Optional[Union[torch.Tensor, np.ndarray]] = None,
    mu_labels: Optional[Sequence[str]] = None,
    h_history: Optional[Union[torch.Tensor, np.ndarray]] = None,
    h_labels: Optional[Sequence[str]] = None,
    h: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    r: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    show_reward_over_time: bool = False,
    bg_pts: Optional[np.ndarray] = None,
    method_name: str = "MPC",
    controlled_label: Optional[str] = None,
    lam: float = 2.0,
    controls_are_velocity: bool = False,
    n_particles: int = 256,
    show_snaps: bool = True,
    snap_times: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    lim: float = 3.5,
    device: Optional[torch.device] = None,
    rng_seed: int = 42,
    show: bool = True,
) -> Tuple[float, float, Optional[float], Optional[float], list]:
    """Visualize controlled vs uncontrolled trajectories and control effort.

    Returns:
        (avg_r_unc, avg_r_ctrl, ctrl_cost_raw, ctrl_cost_obj, figs)
        The ctrl_cost entries are None when ``controls`` is not provided.
        ``figs`` is the list of all matplotlib Figure objects created (in order).
        When ``show=False`` the figures are returned without displaying them,
        which is useful for saving to disk in sweep scripts.

    Optional extra figures (pass the matching argument):
        mu_history  (N, B, C) — Lagrange multiplier evolution (MPC-style).
        h_history   (N, B, C) — per-step constraint value at each method's
                    internal terminal target/estimate (x_hat_N* for HardFlow,
                    x_H*/endpoint estimate for MPC) — NOT the realized state.
        h           — if given (alongside h_history), also evaluates h at the
                    REALIZED trajectory h(x_t) for every t, plotted alongside
                    h(x_hat_N) for comparison.
        show_reward_over_time  — evaluate r along traj_ctrl and traj_unc.
    """
    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    traj_ctrl_np = _to_np(traj_ctrl)
    traj_unc_np  = _to_np(traj_unc)
    x0_np        = _to_np(x0)
    xN_ctrl_np   = _to_np(xN_ctrl)
    xN_unc_np    = _to_np(xN_unc)

    if bg_pts is None:
        bg_pts = sample_target(4000)

    if controlled_label is None:
        controlled_label = f"Controlled ({method_name})"

    rng_viz = np.random.default_rng(rng_seed)
    idx_show = rng_viz.choice(
        traj_ctrl_np.shape[1],
        size=min(n_particles, traj_ctrl_np.shape[1]),
        replace=False,
    )

    avg_r_unc = avg_r_ctrl = None
    if r is not None:
        if device is None:
            device = torch.device("cpu")
        avg_r_unc = r(torch.tensor(xN_unc_np, device=device)).mean().item()
        avg_r_ctrl = r(torch.tensor(xN_ctrl_np, device=device)).mean().item()

    figs: list = []

    # ── Figure 1: trajectory spaghetti ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True,
                             gridspec_kw={"wspace": 0.05})

    for ax, traj_data, pts_end, label, color in [
        (axes[0], traj_unc_np,  xN_unc_np,  "Uncontrolled",        "steelblue"),
        (axes[1], traj_ctrl_np, xN_ctrl_np, controlled_label,      "tomato"),
    ]:
        ax.scatter(bg_pts[:, 0], bg_pts[:, 1], s=2, alpha=0.10, c="gray", zorder=0)
        for i in idx_show:
            ax.plot(traj_data[:, i, 0], traj_data[:, i, 1],
                    lw=0.45, alpha=0.30, color=color)
        ax.scatter(x0_np[idx_show, 0], x0_np[idx_show, 1],
                   s=14, c="slategray", zorder=5, label="$t=0$")
        ax.scatter(pts_end[idx_show, 0], pts_end[idx_show, 1],
                   s=14, c=color, zorder=6, label="$t=1$")
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8, loc="upper left")

    if avg_r_unc is not None:
        fig.suptitle(
            f"{method_name} Trajectories  |  avg r: uncontrolled = {avg_r_unc:.3f}  "
            f"controlled = {avg_r_ctrl:.3f}",
            fontsize=12,
        )
    else:
        fig.suptitle(f"{method_name} Trajectories", fontsize=12)
    plt.tight_layout()
    figs.append(fig)
    if show:
        plt.show()

    # ── Figure 2: snapshot grid ───────────────────────────────────────────────
    if show_snaps:
        ncols = len(snap_times)

        fig, axes = plt.subplots(2, ncols, figsize=(3.0 * ncols, 6.2),
                                 gridspec_kw={"hspace": 0.08, "wspace": 0.06})

        for row, (row_label, color, traj_data) in enumerate([
            (controlled_label, "tomato",    traj_ctrl_np),
            ("Uncontrolled",   "steelblue", traj_unc_np),
        ]):
            T_steps = max(traj_data.shape[0] - 1, 1)
            snap_idx = [
                min(round(t * T_steps), traj_data.shape[0] - 1)
                for t in snap_times
            ]
            for col, (sidx, t_val) in enumerate(zip(snap_idx, snap_times)):
                ax = axes[row, col]
                ax.scatter(bg_pts[:, 0], bg_pts[:, 1], s=2, alpha=0.10, c="gray", zorder=0)
                ax.scatter(traj_data[sidx, :, 0], traj_data[sidx, :, 1],
                           s=5, alpha=0.55, color=color)
                if row == 0:
                    ax.set_title(f"$t = {t_val:.2f}$", fontsize=10)
                ax.set_aspect("equal")
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
                ax.tick_params(labelsize=7)
            axes[row, 0].set_ylabel(row_label, fontsize=10)

        fig.suptitle("Snapshot Comparison  (gray = target)", fontsize=12, y=1.01)
        plt.tight_layout()
        figs.append(fig)
        if show:
            plt.show()

    # ── Figure 3: control displacement + integrated cost ─────────────────────
    ctrl_cost_raw = ctrl_cost_obj = None
    if controls is not None:
        controls_np = _to_np(controls)
        N_steps = controls_np.shape[0]
        dt = 1.0 / N_steps
        dx_np = (dt * controls_np) if controls_are_velocity else controls_np
        dx_mag = np.linalg.norm(dx_np, axis=-1)
        t_axis = np.linspace(0.0, 1.0 - dt, N_steps)

        dx_sq_sum = (dx_mag ** 2).sum(axis=0)
        ctrl_cost_raw = float(dx_sq_sum.mean())
        ctrl_cost_obj = float((dx_sq_sum / (2.0 * lam)).mean())

        print(f"[{method_name}]  avg  Σ‖Δx‖²         = {ctrl_cost_raw:.4f}   (raw integrated effort)")
        print(f"[{method_name}]  avg  Σ‖Δx‖²/(2λ)  = {ctrl_cost_obj:.4f}   (control cost in objective, λ={lam})")

        fig, axes = plt.subplots(1, 2, figsize=(13, 3.2), gridspec_kw={"wspace": 0.3})

        ax = axes[0]
        ax.fill_between(t_axis,
                        np.percentile(dx_mag, 10, axis=1),
                        np.percentile(dx_mag, 90, axis=1),
                        alpha=0.25, color="tomato", label="10–90th pct")
        ax.plot(t_axis, dx_mag.mean(axis=1), color="tomato", lw=2,
                label="mean $\\|\\Delta x\\|$")
        ax.set_xlabel("$t$")
        ax.set_ylabel("$\\|\\Delta x(t)\\|$")
        ax.set_title(f"{method_name}: control displacement magnitude over time")
        ax.legend(fontsize=9)

        ax = axes[1]
        ax.hist(dx_sq_sum, bins=_hist_bins(dx_sq_sum), color="tomato",
                alpha=0.7, edgecolor="white")
        ax.axvline(ctrl_cost_raw, color="black", lw=1.5, linestyle="--",
                   label=f"mean = {ctrl_cost_raw:.3f}")
        ax.set_xlabel("$\\sum_j \\|\\Delta x_j\\|^2$")
        ax.set_ylabel("count")
        ax.set_title(f"{method_name}: integrated control effort (per sample)")
        ax.legend(fontsize=9)
        
        plt.tight_layout()
        figs.append(fig)
        if show:
            plt.show()
    # ── Figure 4: Lagrange multipliers over time ─────────────────────────────
    if mu_history is not None:
        mu_np = _to_np(mu_history)  # (N, B, C)
        if mu_np.ndim == 2:
            mu_np = mu_np[..., None]
        N_mu, _, C_mu = mu_np.shape
        dt_mu = 1.0 / N_mu
        t_mu = np.linspace(dt_mu, 1.0, N_mu)

        if mu_labels is None:
            mu_labels_plot = [fr"$\mu_{{{c}}}$" for c in range(C_mu)]
        else:
            mu_labels_plot = list(mu_labels)
            if len(mu_labels_plot) < C_mu:
                mu_labels_plot += [fr"$\mu_{{{c}}}$" for c in range(len(mu_labels_plot), C_mu)]

        colors = plt.cm.tab10(np.linspace(0, 1, max(C_mu, 1)))

        fig, axes = plt.subplots(1, 2, figsize=(13, 3.2), gridspec_kw={"wspace": 0.3})

        ax = axes[0]
        for c in range(C_mu):
            mu_c = mu_np[:, :, c]
            ax.fill_between(
                t_mu,
                np.percentile(mu_c, 10, axis=1),
                np.percentile(mu_c, 90, axis=1),
                alpha=0.20,
                color=colors[c],
            )
            ax.plot(
                t_mu,
                mu_c.mean(axis=1),
                color=colors[c],
                lw=2,
                label=f"mean {mu_labels_plot[c]}",
            )
        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$\mu(t)$")
        ax.set_title(f"{method_name}: Lagrange multipliers over time")
        ax.legend(fontsize=9)

        ax = axes[1]
        mu_final_np = mu_np[-1]
        for c in range(C_mu):
            ax.hist(
                mu_final_np[:, c],
                bins=_hist_bins(mu_final_np[:, c]),
                alpha=0.55,
                color=colors[c],
                edgecolor="white",
                label=mu_labels_plot[c],
            )
        ax.set_xlabel(r"$\mu$ at final step")
        ax.set_ylabel("count")
        ax.set_title(f"{method_name}: final multiplier distribution")
        ax.legend(fontsize=9)

        plt.tight_layout()
        figs.append(fig)
        if show:
            plt.show()

    # ── Figure 5: constraint values over time (h_history) ────────────────────
    if h_history is not None:
        h_np = _to_np(h_history)            # (N, B, C)
        if h_np.ndim == 2:
            h_np = h_np[..., None]
        N_h, _, C_h = h_np.shape
        t_h = np.linspace(1.0 / N_h, 1.0, N_h)

        h_labels_plot = list(h_labels) if h_labels is not None else []
        if len(h_labels_plot) < C_h:
            h_labels_plot += [f"$h_{{{c}}}$" for c in range(len(h_labels_plot), C_h)]

        colors_h = plt.cm.tab10(np.linspace(0, 1, max(C_h, 1)))

        # Realized h(x_t) along the actual trajectory, if h is provided.
        h_real_np = None
        if h is not None:
            if device is None:
                device = torch.device("cpu")
            traj_ctrl_np_ = _to_np(traj_ctrl)   # (T+1, B, D)
            h_real = []
            for step in traj_ctrl_np_:
                with torch.no_grad():
                    hv = h(torch.tensor(step, device=device)).cpu().numpy()
                if hv.ndim == 1:
                    hv = hv[..., None]
                h_real.append(hv)
            h_real_np = np.stack(h_real, axis=0)   # (T+1, B, C)
            t_real = np.linspace(0.0, 1.0, h_real_np.shape[0])

        fig, axes = plt.subplots(1, 2, figsize=(13, 3.2), gridspec_kw={"wspace": 0.3})

        ax = axes[0]
        for c in range(C_h):
            h_c = h_np[:, :, c]
            ax.fill_between(
                t_h,
                np.percentile(h_c, 10, axis=1),
                np.percentile(h_c, 90, axis=1),
                alpha=0.20, color=colors_h[c],
            )
            ax.plot(t_h, h_c.mean(axis=1), lw=2, color=colors_h[c],
                    label=f"mean {h_labels_plot[c]}  [target $\\hat{{x}}_N$]")
            if h_real_np is not None:
                ax.plot(t_real, h_real_np[:, :, c].mean(axis=1), lw=1.5, ls="--",
                        color=colors_h[c], alpha=0.85,
                        label=f"mean {h_labels_plot[c]}  [realized $x_t$]")
        ax.axhline(0, color="k", lw=1, linestyle="--", label="$h=0$ boundary")
        ax.set_xlabel("$t$")
        ax.set_ylabel("$h$  (solid: target $\\hat{x}_N$;  dashed: realized $x_t$)")
        ax.set_title(f"{method_name}: constraint value over time")
        ax.legend(fontsize=7.5)

        ax = axes[1]
        for c in range(C_h):
            viol = (h_np[:, :, c] > 0).mean(axis=1)
            ax.plot(t_h, viol, lw=2, color=colors_h[c], label=f"{h_labels_plot[c]}  [target]")
            if h_real_np is not None:
                viol_real = (h_real_np[:, :, c] > 0).mean(axis=1)
                ax.plot(t_real, viol_real, lw=1.5, ls="--", color=colors_h[c], alpha=0.85,
                        label=f"{h_labels_plot[c]}  [realized]")
        ax.axhline(0, color="k", lw=1, linestyle="--")
        ax.set_xlabel("$t$")
        ax.set_ylabel("violation fraction")
        ax.set_title(f"{method_name}: fraction of particles violating each constraint")
        ax.legend(fontsize=7.5)

        plt.tight_layout()
        figs.append(fig)
        if show:
            plt.show()

    # ── Figure 6: reward over time ────────────────────────────────────────────
    if show_reward_over_time and r is not None:
        if device is None:
            device = torch.device("cpu")
        traj_ctrl_np_ = _to_np(traj_ctrl)   # (T+1, B, D)
        traj_unc_np_  = _to_np(traj_unc)

        def _eval_r_traj(traj_np):
            rewards = []
            for step in traj_np:
                with torch.no_grad():
                    rv = r(torch.tensor(step, device=device)).cpu().numpy()
                rewards.append(rv)
            return np.stack(rewards, axis=0)  # (T+1, B)

        r_ctrl = _eval_r_traj(traj_ctrl_np_)
        r_unc  = _eval_r_traj(traj_unc_np_)
        T1 = r_ctrl.shape[0]
        t_r = np.linspace(0.0, 1.0, T1)

        fig, axes = plt.subplots(1, 2, figsize=(13, 3.2), gridspec_kw={"wspace": 0.3})

        ax = axes[0]
        for vals, label, color in [
            (r_unc,  "Uncontrolled", "steelblue"),
            (r_ctrl, controlled_label or f"Controlled ({method_name})", "tomato"),
        ]:
            ax.fill_between(t_r,
                            np.percentile(vals, 10, axis=1),
                            np.percentile(vals, 90, axis=1),
                            alpha=0.20, color=color)
            ax.plot(t_r, vals.mean(axis=1), lw=2, color=color, label=label)
        ax.set_xlabel("$t$")
        ax.set_ylabel("$r(x_t)$")
        ax.set_title(f"{method_name}: reward along trajectory")
        ax.legend(fontsize=9)

        ax = axes[1]
        for vals, label, color in [
            (r_unc[:, :],  "Uncontrolled", "steelblue"),
            (r_ctrl[:, :], controlled_label or f"Controlled ({method_name})", "tomato"),
        ]:
            ax.hist(vals[-1], bins=_hist_bins(vals[-1]), alpha=0.55, color=color,
                    edgecolor="white", label=f"{label}  μ={vals[-1].mean():.3f}")
        ax.set_xlabel("$r(x_N)$")
        ax.set_ylabel("count")
        ax.set_title(f"{method_name}: terminal reward distribution")
        ax.legend(fontsize=9)

        plt.tight_layout()
        figs.append(fig)
        if show:
            plt.show()

    return avg_r_unc, avg_r_ctrl, ctrl_cost_raw, ctrl_cost_obj, figs