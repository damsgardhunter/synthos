#!/usr/bin/env python3
"""
Anderson (Pulay) acceleration for the DMFT self-consistency loop.

Linear mixing  Σ_{n+1} = α Σ_n + (1-α) Σ_solver  is the simplest stabilizer,
but it can be very slow for strongly correlated systems near a Mott transition
or for cuprates / nickelates with sharp Fermi-surface features. Anderson
acceleration uses information from the last K iterations to extrapolate to
a better next iterate, typically converging 2-3× faster.

The algorithm (Anderson Type II / Pulay DIIS):
  - Keep history (x_k, F(x_k)) for the last K+1 iterations.
  - Residual r_k = F(x_k) - x_k.  Fixed point ⇔ r = 0.
  - Find coefficients α_k with Σ α_k = 1 minimizing ||Σ α_k r_k||².
  - Mix:  x_{n+1} = Σ α_k x_k + β (Σ α_k r_k)
                  = x_{m-1} + β r_{m-1} + (ΔX + β ΔR) γ
    where γ is the least-squares solution from the constraint elimination.

References:
  D.G. Anderson, JACM 12, 547 (1965)
  P. Pulay, Chem. Phys. Lett. 73, 393 (1980)
  H.F. Walker & P. Ni, SIAM J. Numer. Anal. 49, 1715 (2011)
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional


class AndersonMixer:
    """
    Anderson-accelerated fixed-point mixer.

    Drop-in replacement for linear mixing:
        # Linear:
        sigma = alpha * sigma_old + (1 - alpha) * sigma_new

        # Anderson:
        sigma = mixer.update(sigma_old, sigma_new)

    The first iteration falls back to linear mixing with weight `beta`.
    For numerical safety, the mixer:
      - Restarts history if residual norm increases sharply (instability)
      - Falls back to linear mixing if least-squares is degenerate
      - Periodically restarts after `restart_iter` iterations (anti-stagnation)
    """

    def __init__(
        self,
        history_depth: int = 5,
        beta: float = 1.0,
        regularization: float = 1e-10,
        restart_iter: int = 30,
        linear_fallback: float = 0.3,
        residual_blowup_factor: float = 5.0,
    ):
        """
        Args:
            history_depth: number of previous iterations to use (K). Typical 3-8.
                Larger K = faster convergence but more memory and potential
                ill-conditioning. K=5 is a good default.
            beta: Anderson mixing parameter. β=1 is "full Anderson" (most
                aggressive). β<1 damps. Use β=1.0 for well-behaved problems,
                β=0.5 near Mott transitions.
            regularization: SVD cutoff for the least-squares solve (Tikhonov).
                Larger → more stable but slower convergence.
            restart_iter: hard reset of history every N iterations. Prevents
                stagnation when stuck on a saddle.
            linear_fallback: mixing weight α used when Anderson fails or for
                the very first iteration. x_{n+1} = α x_new + (1-α) x_old.
                Convention: high α = trust solver more.
            residual_blowup_factor: if ||r_n|| > factor * min(||r_k||), the
                mixer suspects instability and clears history.
        """
        self.K = int(history_depth)
        self.beta = float(beta)
        self.reg = float(regularization)
        self.restart_iter = int(restart_iter)
        self.linear_fallback = float(linear_fallback)
        self.residual_blowup_factor = float(residual_blowup_factor)

        self.x_history: List[np.ndarray] = []
        self.f_history: List[np.ndarray] = []
        self.residual_norms: List[float] = []
        self.iter_since_restart = 0
        self.total_iter = 0
        self.last_method = "init"

    # ── State management ────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all history. Use when entering a new regime (T, U, etc.)."""
        self.x_history.clear()
        self.f_history.clear()
        self.residual_norms.clear()
        self.iter_since_restart = 0

    def _linear_mix(self, x_in: np.ndarray, f_x: np.ndarray) -> np.ndarray:
        """Fallback linear mix with weight self.linear_fallback toward f_x."""
        α = self.linear_fallback
        return α * f_x + (1.0 - α) * x_in

    def _detect_blowup(self, r_norm: float) -> bool:
        """Check if the latest residual is much worse than the minimum so far."""
        if len(self.residual_norms) < 3:
            return False
        recent_min = min(self.residual_norms[-self.K:]) if self.residual_norms else r_norm
        return recent_min > 0 and r_norm > self.residual_blowup_factor * recent_min

    # ── Main update ─────────────────────────────────────────────────────

    def update(self, x_in: np.ndarray, f_x: np.ndarray) -> np.ndarray:
        """
        Compute the next iterate using Anderson acceleration.

        Args:
            x_in: current input to the fixed-point map (e.g., Σ_n)
            f_x:  output F(x_in) from the solver (e.g., Σ_solver from CTHYB)

        Returns:
            x_next: the next iterate to feed back into the solver
        """
        if x_in.shape != f_x.shape:
            raise ValueError(f"Shape mismatch: x_in={x_in.shape} vs f_x={f_x.shape}")

        self.total_iter += 1
        self.iter_since_restart += 1

        # Compute residual norm and check for blowup
        r = f_x - x_in
        r_norm = float(np.linalg.norm(r))
        if self._detect_blowup(r_norm):
            self.reset()
            self.last_method = "reset_blowup"
            self.x_history.append(x_in.copy())
            self.f_history.append(f_x.copy())
            self.residual_norms.append(r_norm)
            return self._linear_mix(x_in, f_x)

        # Periodic restart
        if self.iter_since_restart >= self.restart_iter:
            self.reset()
            self.iter_since_restart = 1

        # Push current iteration into history
        self.x_history.append(x_in.copy())
        self.f_history.append(f_x.copy())
        self.residual_norms.append(r_norm)

        # Trim to K+1 entries
        if len(self.x_history) > self.K + 1:
            self.x_history = self.x_history[-(self.K + 1):]
            self.f_history = self.f_history[-(self.K + 1):]
            self.residual_norms = self.residual_norms[-(self.K + 1):]

        m = len(self.x_history)
        if m < 2:
            self.last_method = "linear_warmup"
            return self._linear_mix(x_in, f_x)

        # Build residual differences (relative to most recent)
        r_hist = [self.f_history[i] - self.x_history[i] for i in range(m)]
        ncol = m - 1
        dim = x_in.size
        dtype = np.result_type(x_in.dtype, f_x.dtype, np.complex64)

        dR = np.empty((dim, ncol), dtype=dtype)
        dX = np.empty((dim, ncol), dtype=dtype)
        for k in range(ncol):
            dR[:, k] = (r_hist[k] - r_hist[-1]).ravel()
            dX[:, k] = (self.x_history[k] - self.x_history[-1]).ravel()

        b = -r_hist[-1].ravel()

        # Regularized least-squares for γ via SVD
        try:
            gamma, _residuals, _rank, _svals = np.linalg.lstsq(
                dR, b, rcond=self.reg,
            )
        except np.linalg.LinAlgError:
            self.last_method = "linalg_error_fallback"
            return self._linear_mix(x_in, f_x)

        # Reject ill-conditioned step (||γ|| huge means ΔR is rank-deficient)
        if not np.all(np.isfinite(gamma)) or np.linalg.norm(gamma) > 1e6:
            self.last_method = "ill_conditioned_fallback"
            return self._linear_mix(x_in, f_x)

        # Anderson update
        x_last = self.x_history[-1].ravel()
        r_last = r_hist[-1].ravel()
        x_next = x_last + self.beta * r_last + (dX + self.beta * dR) @ gamma

        # Cast back to original dtype if possible
        if not np.iscomplexobj(x_in):
            x_next = x_next.real

        self.last_method = f"anderson_K={m-1}"
        return x_next.reshape(x_in.shape).astype(x_in.dtype, copy=False)

    # ── Diagnostics ─────────────────────────────────────────────────────

    def status(self) -> dict:
        """Return a small dict for logging."""
        return {
            "method": self.last_method,
            "iter": self.total_iter,
            "history_depth": len(self.x_history),
            "last_residual_norm": (
                float(self.residual_norms[-1]) if self.residual_norms else None
            ),
            "min_residual_norm": (
                float(min(self.residual_norms)) if self.residual_norms else None
            ),
        }


def make_mixer(
    method: str = "anderson",
    linear_alpha: float = 0.5,
    history_depth: int = 5,
    beta: float = 1.0,
):
    """
    Factory for the mixer to use in a DMFT loop.

    Args:
        method: "anderson" (default, accelerated) or "linear" (legacy).
        linear_alpha: weight of x_old in linear mode: x = α x_old + (1-α) x_new.
            Ignored for Anderson (use `beta` instead).
        history_depth: Anderson history size K (only Anderson).
        beta: Anderson step size (only Anderson).

    Returns:
        Either an AndersonMixer or a LinearMixer (mimicking the same .update API).
    """
    if method == "linear":
        return LinearMixer(alpha=linear_alpha)
    if method == "anderson":
        # Linear-fallback uses 1 - linear_alpha so "alpha" convention matches:
        # linear: x = α x_old + (1-α) x_new  → trust toward x_new = (1-α).
        return AndersonMixer(
            history_depth=history_depth,
            beta=beta,
            linear_fallback=max(0.1, min(0.9, 1.0 - linear_alpha)),
        )
    raise ValueError(f"Unknown mixing method: {method!r}")


class LinearMixer:
    """Legacy linear mixer with the same .update API as AndersonMixer."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = float(alpha)
        self.total_iter = 0
        self.last_method = "linear"

    def reset(self) -> None:
        self.total_iter = 0

    def update(self, x_in: np.ndarray, f_x: np.ndarray) -> np.ndarray:
        self.total_iter += 1
        return self.alpha * x_in + (1.0 - self.alpha) * f_x

    def status(self) -> dict:
        return {"method": "linear", "iter": self.total_iter, "alpha": self.alpha}
