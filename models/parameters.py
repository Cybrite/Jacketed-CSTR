"""Process parameter definitions for the jacketed exothermic CSTR.

The values are chosen to be representative of the classic exothermic CSTR
benchmark used in advanced process control texts. Units are kept consistent in
engineering-style cal, g, L, min, and K units to avoid hidden conversion
errors inside the dynamic equations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class CSTRParameters:
    """Physical and kinetic parameters for the CSTR model."""

    V: float = 100.0
    Vj: float = 20.0
    F: float = 100.0
    Fc: float = 100.0
    Ca0: float = 1.0
    T0: float = 350.0
    Tcin0: float = 300.0
    UA: float = 5.0e4
    deltaH: float = -5.0e4
    E: float = 1.738625e4
    R: float = 1.987
    k0: float = 7.2e10
    rho: float = 1000.0
    Cp: float = 0.239
    rho_j: float = 1000.0
    Cp_j: float = 1.0
    Fc_min: float = 20.0
    Fc_max: float = 180.0
    T_safe: float = 500.0
    Ca_min: float = 0.0
    T_min: float = 250.0
    Fc_bias: float = 100.0
    dt: float = 0.1
    episode_steps: int = 200

    def as_dict(self) -> Dict[str, float]:
        """Return the parameters as a standard Python dictionary."""

        return asdict(self)

    @property
    def Tc0(self) -> float:
        return self.Tcin0

    @property
    def F_min(self) -> float:
        return self.Fc_min

    @property
    def F_max(self) -> float:
        return self.Fc_max

    @property
    def F_bias(self) -> float:
        return self.Fc_bias



def default_parameters() -> CSTRParameters:
    """Return the default textbook parameter set."""

    return CSTRParameters()