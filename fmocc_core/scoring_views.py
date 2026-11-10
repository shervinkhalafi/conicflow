"""
scoring_views.py -- random views of a decoded image, for constraint scoring.

A CLIP-family scorer evaluated on the decoded image itself can be satisfied by a
pixel-aligned, high-frequency pattern: the score rises by many points while the
image barely changes. Averaging the score over random views (crop, shift, flip,
low-pass, noise) removes that route, because such a pattern does not survive the
views while a change in content does. Measured 2026-11-09 (results/edit_diag): on
a pure high-frequency pattern the views keep 21% of the rms, on a low-frequency
one all of it; as the constraint scorer it was the one change that turned the
suite configuration into one that makes the requested edit.

Used by run_sweep's --sweep_constraint_views, edit_diag.py and pie_diag.py.
"""
import torch
import torch.nn.functional as F


def augment(img: torch.Tensor, K: int, gen=None, identity: bool = True) -> torch.Tensor:
    """(1, 3, S, S) in [-1, 1] -> (K, 3, S, S): the image itself (when `identity`)
    plus random views -- crop keeping 60-100% of the side, shift, horizontal flip,
    low-pass down to 96..S px, noise 0.02. Differentiable; returns fp32.

    `gen`: a torch.Generator on img's device for reproducible views; None draws
    from the global generator.
    """
    x = img.float()
    S, dev = x.shape[-1], x.device
    views = [x] if identity else []
    n = K - len(views)
    if n > 0:
        rnd = lambda k: torch.rand(k, device=dev, generator=gen)
        scale = 0.6 + 0.4 * rnd(n)
        theta = torch.zeros(n, 2, 3, device=dev)
        theta[:, 0, 0] = scale * ((rnd(n) < 0.5).float() * 2 - 1)
        theta[:, 1, 1] = scale
        theta[:, 0, 2] = (1 - scale) * (2 * rnd(n) - 1)
        theta[:, 1, 2] = (1 - scale) * (2 * rnd(n) - 1)
        grid = F.affine_grid(theta, (n, 3, S, S), align_corners=False)
        v = F.grid_sample(x.expand(n, -1, -1, -1), grid, mode="bilinear",
                          padding_mode="reflection", align_corners=False)
        outs = []
        for i in range(n):
            s = int(96 + (S - 96) * rnd(1).item())
            vi = F.interpolate(v[i:i + 1], size=(s, s), mode="bilinear", antialias=True, align_corners=False)
            outs.append(F.interpolate(vi, size=(S, S), mode="bilinear", align_corners=False))
        v = torch.cat(outs)
        views.append(v + 0.02 * torch.randn(v.shape, device=dev, generator=gen))
    return torch.cat(views).clamp(-1, 1)
