"""Small OSQP wrapper used by the Phase D safety layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

try:
    import osqp
except ImportError as exc:  # pragma: no cover
    osqp = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass(frozen=True)
class QPSolution:
    """QP solve result."""

    primal: np.ndarray | None
    status: str
    solve_time: float = 0.0
    iterations: int = 0


class OSQPQPSolver:
    """Solve small convex QPs with deterministic OSQP settings."""

    def __init__(self) -> None:
        if osqp is None:
            raise RuntimeError("OSQP is required for the Phase D safety filter.") from _IMPORT_ERROR

    def solve(
        self,
        *,
        quadratic_cost: np.ndarray,
        linear_cost: np.ndarray,
        constraint_matrix: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> QPSolution:
        """Solve the box/affine constrained QP."""

        solver = osqp.OSQP()
        solver.setup(
            P=sparse.csc_matrix(quadratic_cost),
            q=linear_cost,
            A=sparse.csc_matrix(constraint_matrix),
            l=lower_bounds,
            u=upper_bounds,
            verbose=False,
            polishing=True,
            adaptive_rho=True,
            adaptive_rho_interval=25,
            eps_abs=1e-6,
            eps_rel=1e-6,
            max_iter=20_000,
        )
        result = solver.solve(raise_error=False)
        status = result.info.status.lower()
        solve_time = float(getattr(result.info, "run_time", 0.0) or 0.0)
        if not np.isfinite(solve_time):
            solve_time = 0.0
        iterations = int(getattr(result.info, "iter", 0) or 0)
        if status not in {"solved", "solved inaccurate"}:
            return QPSolution(primal=None, status=status, solve_time=solve_time, iterations=iterations)
        if result.x is None or not np.all(np.isfinite(result.x)):
            return QPSolution(
                primal=None,
                status=f"{status}_nan",
                solve_time=solve_time,
                iterations=iterations,
            )
        return QPSolution(
            primal=np.asarray(result.x, dtype=float),
            status=status,
            solve_time=solve_time,
            iterations=iterations,
        )
