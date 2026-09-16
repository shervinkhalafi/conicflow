import torch
import numpy as np
from typing import Callable, Optional, Sequence, Tuple, Union


def _center_scale(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(0)
    return X / (X.std(0) + 1e-8)


def sample_two_moons(
    n_samples: int,
    *,
    noise: float = 0.05,
    return_labels: bool = False,
    scale: float = 0.75,
):
    """Match common.datasets.sample_two_moons without importing JAX/TF."""
    from sklearn.datasets import make_moons

    samples, labels = make_moons(n_samples=n_samples, noise=noise, random_state=None)
    samples = samples.astype(np.float32)
    samples = _center_scale(samples)
    samples = scale * samples

    if return_labels:
        return samples, labels
    return samples


# ── helpers ───────────────────────────────────────────────────────────────────
def sample_rho0(n, rescale=1.0, device=None):
    """Same base distribution as training: N(0, rescale^2 I)."""
    return rescale * torch.randn(n, 2, device=device)


@torch.no_grad()
def flow_map_sample(model, x0, N):
    """Apply the flow map in N equal steps: t_0=0 → t_1 → … → t_N=1."""
    ts = torch.linspace(0.0, 1.0, N + 1, device=x0.device)
    x = x0.clone()
    for i in range(N):
        s = ts[i].expand(x.shape[0])
        t = ts[i + 1].expand(x.shape[0])
        x = model(s, t, x)
    return x


def make_target(n_plot=5_000, n_total=500_000, noise=0.05):
    """Use the exact training sampler, then subsample for plotting."""
    all_samples = sample_two_moons(n_total, noise=noise)
    idx = np.random.choice(n_total, size=n_plot, replace=False)
    return all_samples[idx]


def sample_target(n_plot=5_000, n_total=500_000, noise=0.05):
    """Alias for make_target."""
    return make_target(n_plot=n_plot, n_total=n_total, noise=noise)




@torch.no_grad()
def euler_sample(
    model,
    n: int = 2000,
    n_steps: int = 100,
    return_traj: bool = False,
    device: Optional[torch.device] = None,
    rescale: Optional[float] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Euler integration of the induced velocity field b(t, x) = phi(t, t, x)
    from t=0 to t=1. Base samples match training: N(0, rescale^2 I).
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    if rescale is None:
        rescale = float(getattr(model, "rescale", 1.0))
    x = rescale * torch.randn(n, 2, device=device)
    dt = 1.0 / n_steps
    ts = torch.linspace(0.0, 1.0 - dt, n_steps, device=device)

    traj = [x.cpu()] if return_traj else None

    for t_val in ts:
        v = model.calc_b(t_val, x)   # b(t,x) = phi(t,t,x)
        x = x + dt * v
        if return_traj:
            traj.append(x.cpu())

    if return_traj:
        return x.cpu(), torch.stack(traj)   # (n_steps+1, n, 2)
    return x.cpu()


