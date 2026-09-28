"""Butcher tables used by the legacy SBGK Sod solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ButcherTable:
    """Explicit/implicit RK coefficients from getButcher.m.

    Matrix coefficients are stored as two-dimensional arrays. MATLAB column
    vectors be/ce/bi/ci are normalized to one-dimensional arrays for Python
    stage loops.
    """

    Ae: np.ndarray
    be: np.ndarray
    ce: np.ndarray
    Ai: np.ndarray
    bi: np.ndarray
    ci: np.ndarray

    @property
    def num_stages(self) -> int:
        return self.be.size


def get_butcher(table_number: int) -> ButcherTable:
    """Return the supported legacy SBGK Butcher table."""
    if table_number == 7:
        return _make_table(
            Ae=[[0.0]],
            be=[1.0],
            ce=[0.0],
            Ai=[[1.0]],
            bi=[1.0],
            ci=[1.0],
        )

    if table_number == 2:
        gam = 1.0 - 1.0 / np.sqrt(2.0)
        return _make_table(
            Ae=[[0.0, 0.0], [1.0, 0.0]],
            be=[0.5, 0.5],
            ce=[0.0, 1.0],
            Ai=[[gam, 0.0], [1.0 - 2.0 * gam, gam]],
            bi=[0.5, 0.5],
            ci=[gam, 1.0 - gam],
        )

    raise ValueError(f"Unsupported SBGK Butcher table number: {table_number}")


def _make_table(
    *,
    Ae: list[list[float]],
    be: list[float],
    ce: list[float],
    Ai: list[list[float]],
    bi: list[float],
    ci: list[float],
) -> ButcherTable:
    return ButcherTable(
        Ae=np.asarray(Ae, dtype=float),
        be=np.asarray(be, dtype=float),
        ce=np.asarray(ce, dtype=float),
        Ai=np.asarray(Ai, dtype=float),
        bi=np.asarray(bi, dtype=float),
        ci=np.asarray(ci, dtype=float),
    )
