"""Functional LIF / ALIF dynamics and spike surrogates for the e-prop core.

Conventions (identical to the legacy ``LIF_Node``):

    v_t = alpha * v_{t-1} * (1 - z_{t-1}) + I_t          multiplicative reset
    z_t = Theta(v_t - A_t)

with ``A_t = thresh`` for LIF and ``A_t = thresh + beta * a_t`` for ALIF,
``a_{t+1} = rho * a_t + z_t``.

The e-prop pseudo-derivative is the legacy triangle

    h_t = gain * max(0, 1 - |v_t - A_t| / thresh)

and is a plain tensor expression (no autograd). For BPTT the spike is a
``torch.autograd.Function`` whose backward is either the legacy boxcar
(``|v - A| < 0.1``) or the same triangle.
"""
from __future__ import annotations

import torch
from torch import Tensor


# ----------------------------------------------------------------------------
# surrogate spike functions (autograd) -- used only when training with BPTT
# ----------------------------------------------------------------------------
class BoxcarSpike(torch.autograd.Function):
    """Heaviside forward, boxcar surrogate of half-width ``hw`` around A.
    Same arithmetic as legacy ``neurons.Boxcar`` (subthresh 0.1)."""

    @staticmethod
    def forward(ctx, v: Tensor, A, hw: float):
        ctx.save_for_backward(v)
        ctx.A = A
        ctx.hw = hw
        return v.gt(A).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        g = grad_out * ((v - ctx.A).abs() < ctx.hw).float()
        # d z / d A = -d z / d v : propagate into an adaptive threshold tensor (ALIF)
        gA = -g if isinstance(ctx.A, torch.Tensor) else None
        return g, gA, None


class TriangleSpike(torch.autograd.Function):
    """Heaviside forward, triangular surrogate ``gain * relu(1 - |v-A|/thresh)``."""

    @staticmethod
    def forward(ctx, v: Tensor, A, thresh: float, gain: float):
        ctx.save_for_backward(v)
        ctx.A, ctx.thresh, ctx.gain = A, thresh, gain
        return v.gt(A).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        g = grad_out * ctx.gain * torch.clamp(1 - (v - ctx.A).abs() / ctx.thresh, min=0.0)
        gA = -g if isinstance(ctx.A, torch.Tensor) else None
        return g, gA, None, None


def spike(v: Tensor, A, thresh: float, surrogate: str = "boxcar",
          gain: float = 0.6, boxcar_halfwidth: float = 0.1) -> Tensor:
    """Spike with a surrogate gradient (``surrogate`` in {'boxcar','triangle'})."""
    if surrogate == "boxcar":
        return BoxcarSpike.apply(v, A, boxcar_halfwidth)
    if surrogate == "triangle":
        return TriangleSpike.apply(v, A, thresh, gain)
    raise ValueError(surrogate)


# ----------------------------------------------------------------------------
# pure functional pieces (no autograd needed for e-prop)
# ----------------------------------------------------------------------------
def membrane_step(v: Tensor, z_prev: Tensor, alpha: float, I: Tensor,
                  detach_reset: bool = False) -> Tensor:
    """Legacy LIF membrane update with multiplicative reset.

    ``detach_reset=True`` computes the same value as ``v*alpha*(1-z_prev)+I`` but
    treats the reset amount ``v*z_prev`` as a constant, so autograd flows through
    ``v`` undiminished -- the e-prop convention (the eligibility trace
    ``eps_v <- alpha*eps_v + x`` ignores the reset path)."""
    if detach_reset:
        return (v - (v * z_prev).detach()) * alpha + I
    return v * alpha * (1 - z_prev) + I


def pseudo_derivative(v: Tensor, A, thresh: float, gain: float) -> Tensor:
    """Triangle pseudo-derivative ``gain * max(0, 1 - |v - A| / thresh)``.

    ``A`` may be a float (LIF) or a tensor (ALIF effective threshold)."""
    return gain * torch.max(torch.zeros_like(v), 1 - torch.abs((v - A) / thresh))


def alif_threshold(thresh: float, beta: float, a: Tensor):
    """Effective threshold ``A_t = thresh + beta * a_t`` (returns float if beta==0)."""
    if beta == 0.0:
        return thresh
    return thresh + beta * a


def alif_adapt(a: Tensor, z: Tensor, rho: float) -> Tensor:
    """``a_{t+1} = rho * a_t + z_t``."""
    return rho * a + z


def eligibility_lif(h: Tensor, eps_v: Tensor) -> Tensor:
    """LIF eligibility ``e_{ji} = h_j * eps_v_i`` -> (B, post, pre)."""
    return torch.einsum("br,bi->bri", h, eps_v)


def eligibility_alif(h: Tensor, eps_v: Tensor, eps_a: Tensor, beta: float, rho: float):
    """Full ALIF eligibility (Bellec et al. 2020, eqs. 23-25 with the legacy
    trace conventions):

        e_{ji}^t       = h_j^t * (eps_v_i^t - beta * eps_a_{ji}^t)
        eps_a_{ji}^{t+1} = h_j^t * eps_v_i^t + (rho - h_j^t * beta) * eps_a_{ji}^t

    Returns ``(e, eps_a_next)``; both (B, post, pre)."""
    hv = torch.einsum("br,bi->bri", h, eps_v)
    e = hv - beta * h.unsqueeze(2) * eps_a
    eps_a_next = hv + (rho - beta * h).unsqueeze(2) * eps_a
    return e, eps_a_next
