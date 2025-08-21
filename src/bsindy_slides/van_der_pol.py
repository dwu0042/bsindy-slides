"""SINDY Example with the Van der Pol oscillator"""
from typing import Iterable

import numpy as np
from numpy import typing as npt
from scipy import integrate
import pysindy as pys

def van_der_pol_oscillator_1144(
        t: npt.DTypeLike, 
        x: npt.NDArray
) -> npt.NDArray:
    return np.array([
        x[1],
        -x[0] + 4*x[1] - 4*x[0]**2*x[1],
    ])

DENSE = 1001
SPARSE = 49

def generate_solution(
    integration_limits: Iterable[float] = (0, 12),
    initial_state: Iterable[float] = (2, 0),
    span_density: int = DENSE,
):
    int_limit_lower, int_limit_upper = integration_limits
    dense_time_span = np.linspace(int_limit_lower, int_limit_upper, span_density)
    
    state_space_solution = integrate.solve_ivp(
        van_der_pol_oscillator_1144, 
        t_span=integration_limits,
        y0=initial_state,
        t_eval=dense_time_span
    )

    return state_space_solution

def add_noise(
    base: npt.NDArray,
    std_dev = 0.1,
    rng = None,
):
    if rng is None:
        rng = np.random.default_rng()
    
    noise = rng.normal(loc=0, scale=std_dev, size=base.shape)

    return base + noise

def sindy_fit(
    time: npt.ArrayLike,
    data: npt.ArrayLike,
) -> pys.SINDy:
    model = pys.SINDy(
        feature_names=['x1', 'x2'],
        optimizer=pys.STLSQ(threshold=0.45),
        feature_library=pys.PolynomialLibrary(degree=3),
    )

    model.fit(data, t=time)

    return model


def main(rng=None):

    state_sol = generate_solution(span_density=SPARSE)
    observations = add_noise(state_sol.y, rng=rng)

    model = sindy_fit(state_sol.t, observations.T)

    data = {
        'truth': state_sol, 
        'observations': observations,
    }

    return model, data


if __name__ == "__main__":
    rng = np.random.default_rng(20250911)
    model, _ = main(rng=rng)
    model.print()