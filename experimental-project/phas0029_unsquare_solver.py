#!/usr/bin/env python3
"""PHAS0029 final assignment solver: infinite square and embedded "unsquare" wells.

This standalone script implements:
- RK4 shooting-method integration of the 1D time-independent Schrodinger equation
- Secant-method eigenvalue search for states satisfying psi(+a)=0
- Validation against infinite-square-well analytical energies/eigenfunctions
- Excited-state checks (including high n), convergence mini-study, and secant stress test
- Embedded harmonic and embedded finite-square potentials with plots

Run:
    python3 phas0029_unsquare_solver.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib.pyplot as plt


# Physical constants (SI)
M_ELECTRON = 9.109_383_702e-31
HBAR = 1.054_571_817e-34
E_CHARGE = 1.602_176_634e-19

# Problem geometry
DOT_SIDE = 6.0e-9
HALF_WIDTH = DOT_SIDE / 2.0
DEFAULT_N = 2400


Potential = Callable[[float | np.ndarray], float | np.ndarray]


@dataclass
class EigenstateResult:
    """Container for one numerically solved eigenstate."""

    state_index: int
    energy_j: float
    energy_ev: float
    iterations: int
    endpoint_residual: float
    x: np.ndarray
    psi_raw: np.ndarray
    psi_norm: np.ndarray


def make_grid(a: float, n_points: int) -> np.ndarray:
    """Return an evenly spaced x-grid in [-a, +a]."""

    return np.linspace(-a, a, n_points)


def infinite_well_potential(x: float | np.ndarray) -> float | np.ndarray:
    """Potential inside the benchmark infinite square well interior: V(x)=0."""

    x_arr = np.asarray(x)
    values = np.zeros_like(x_arr, dtype=float)
    return float(values) if np.isscalar(x) else values


def make_harmonic_embedded_potential(a: float, v0: float) -> Potential:
    """Return V(x)=V0*(x/a)^2 embedded within the same outer infinite walls."""

    def potential(x: float | np.ndarray) -> float | np.ndarray:
        x_arr = np.asarray(x)
        values = v0 * (x_arr / a) ** 2
        return float(values) if np.isscalar(x) else values

    return potential


def make_embedded_finite_square_potential(
    a: float,
    v0: float,
    inner_half_width: float | None = None,
) -> Potential:
    """Return the embedded finite-square profile with central low-potential region."""

    inner = a / 2.0 if inner_half_width is None else inner_half_width

    def potential(x: float | np.ndarray) -> float | np.ndarray:
        x_arr = np.asarray(x)
        values = np.where(np.abs(x_arr) <= inner, 0.0, v0)
        return float(values) if np.isscalar(x) else values

    return potential


def schrodinger_rhs(state: np.ndarray, x: float, energy: float, potential_fn: Potential) -> np.ndarray:
    """Return RHS of coupled first-order Schrodinger shooting equations."""

    psi, phi = state
    v_x = float(potential_fn(x))
    dpsi_dx = phi
    dphi_dx = (2.0 * M_ELECTRON / HBAR**2) * (v_x - energy) * psi
    return np.array([dpsi_dx, dphi_dx], dtype=float)


def integrate_wavefunction_rk4(
    energy: float,
    x_grid: np.ndarray,
    potential_fn: Potential,
    psi_left: float = 0.0,
    phi_left: float = 1.0,
) -> np.ndarray:
    """Integrate the Schrodinger system from -a to +a with RK4 and return psi(x)."""

    psi, _ = integrate_state_rk4(
        energy=energy,
        x_grid=x_grid,
        potential_fn=potential_fn,
        psi_left=psi_left,
        phi_left=phi_left,
    )
    return psi


def integrate_state_rk4(
    energy: float,
    x_grid: np.ndarray,
    potential_fn: Potential,
    psi_left: float = 0.0,
    phi_left: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the Schrodinger system from start(x_grid) and return (psi, phi)."""

    psi = np.empty_like(x_grid)
    phi = np.empty_like(x_grid)
    psi[0] = psi_left
    phi[0] = phi_left

    for idx in range(x_grid.size - 1):
        x_i = x_grid[idx]
        h = x_grid[idx + 1] - x_i
        state = np.array([psi[idx], phi[idx]], dtype=float)

        k1 = h * schrodinger_rhs(state, x_i, energy, potential_fn)
        k2 = h * schrodinger_rhs(state + 0.5 * k1, x_i + 0.5 * h, energy, potential_fn)
        k3 = h * schrodinger_rhs(state + 0.5 * k2, x_i + 0.5 * h, energy, potential_fn)
        k4 = h * schrodinger_rhs(state + k3, x_i + h, energy, potential_fn)

        next_state = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        psi[idx + 1] = next_state[0]
        phi[idx + 1] = next_state[1]

    return psi, phi


def endpoint_residual(energy: float, x_grid: np.ndarray, potential_fn: Potential) -> float:
    """Return g(E)=psi(+a;E) for shooting root finding."""

    psi = integrate_wavefunction_rk4(energy, x_grid, potential_fn)
    return float(psi[-1])


def secant_search_eigenvalue(
    energy_1: float,
    energy_2: float,
    x_grid: np.ndarray,
    potential_fn: Potential,
    energy_tol: float,
    residual_tol: float = 1e-10,
    max_iter: int = 120,
) -> tuple[float, np.ndarray, int, float]:
    """Use secant iterations to find an eigenenergy where psi(+a)=0."""

    e_prev = energy_1
    e_curr = energy_2
    g_prev = endpoint_residual(e_prev, x_grid, potential_fn)
    g_curr = endpoint_residual(e_curr, x_grid, potential_fn)

    if not (np.isfinite(g_prev) and np.isfinite(g_curr)):
        raise RuntimeError("Initial secant residuals are not finite.")

    has_bracket = (g_prev == 0.0) or (g_curr == 0.0) or (np.signbit(g_prev) != np.signbit(g_curr))
    if has_bracket:
        left_e, left_g = (e_prev, g_prev)
        right_e, right_g = (e_curr, g_curr)
        if left_e > right_e:
            left_e, right_e = right_e, left_e
            left_g, right_g = right_g, left_g
    else:
        left_e = right_e = left_g = right_g = 0.0

    for iteration in range(1, max_iter + 1):
        denominator = g_curr - g_prev
        if abs(denominator) < 1e-30:
            if has_bracket:
                e_next = 0.5 * (left_e + right_e)
            else:
                raise RuntimeError("Secant denominator is effectively zero.")
        else:
            e_next = e_curr - g_curr * (e_curr - e_prev) / denominator
            if has_bracket and not (left_e < e_next < right_e):
                e_next = 0.5 * (left_e + right_e)

        g_next = endpoint_residual(e_next, x_grid, potential_fn)

        if not np.isfinite(g_next):
            if has_bracket:
                e_next = 0.5 * (left_e + right_e)
                g_next = endpoint_residual(e_next, x_grid, potential_fn)
            else:
                raise RuntimeError("Secant iteration produced non-finite residual.")

        converged_energy = abs(e_next - e_curr) < energy_tol
        converged_endpoint = abs(g_next) < residual_tol
        if has_bracket and abs(right_e - left_e) < energy_tol:
            converged_energy = True
        if converged_energy and (converged_endpoint or has_bracket):
            psi = integrate_wavefunction_rk4(e_next, x_grid, potential_fn)
            return float(e_next), psi, iteration, abs(g_next)

        if has_bracket:
            if left_g == 0.0:
                right_e, right_g = e_next, g_next
            elif right_g == 0.0:
                left_e, left_g = e_next, g_next
            elif np.signbit(left_g) != np.signbit(g_next):
                right_e, right_g = e_next, g_next
            else:
                left_e, left_g = e_next, g_next

        e_prev, g_prev = e_curr, g_curr
        e_curr, g_curr = e_next, g_next

    raise RuntimeError("Secant method failed to converge within max_iter.")


def trapezoidal_integral(y: np.ndarray, x: np.ndarray) -> float:
    """Compute integral using the trapezoidal rule on a uniform grid."""

    h = x[1] - x[0]
    return float(h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1]))


def normalize_wavefunction(psi: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Return L2-normalized wavefunction on the finite domain."""

    max_amp = float(np.max(np.abs(psi)))
    if max_amp == 0.0 or not np.isfinite(max_amp):
        raise RuntimeError("Cannot normalize: zero norm wavefunction.")
    scaled = psi / max_amp
    scaled_norm = np.sqrt(trapezoidal_integral(np.abs(scaled) ** 2, x_grid))
    norm = max_amp * scaled_norm
    if norm == 0.0 or not np.isfinite(norm):
        raise RuntimeError("Cannot normalize: non-finite norm wavefunction.")
    return psi / norm


def matched_wavefunction_from_both_sides(
    energy: float,
    state_index: int,
    x_grid: np.ndarray,
    potential_fn: Potential,
) -> np.ndarray:
    """Build a stable full-domain psi(x) by matching left/right shooting solutions at x=0.

    This is useful for stiff embedded potentials where one-sided shooting can leave an
    exponentially amplified contamination near the opposite boundary.
    """

    psi_left, phi_left = integrate_state_rk4(
        energy=energy,
        x_grid=x_grid,
        potential_fn=potential_fn,
        psi_left=0.0,
        phi_left=1.0,
    )

    x_rev = x_grid[::-1]
    psi_right_rev, phi_right_rev = integrate_state_rk4(
        energy=energy,
        x_grid=x_rev,
        potential_fn=potential_fn,
        psi_left=0.0,
        phi_left=-1.0,
    )
    psi_right = psi_right_rev[::-1]
    phi_right = phi_right_rev[::-1]

    mid_idx = x_grid.size // 2
    if state_index % 2 == 1:
        denominator = psi_right[mid_idx]
        scale = 1.0 if abs(denominator) < 1e-30 else psi_left[mid_idx] / denominator
    else:
        denominator = phi_right[mid_idx]
        scale = 1.0 if abs(denominator) < 1e-30 else phi_left[mid_idx] / denominator

    psi_combined = np.empty_like(x_grid)
    psi_combined[: mid_idx + 1] = psi_left[: mid_idx + 1]
    psi_combined[mid_idx + 1 :] = scale * psi_right[mid_idx + 1 :]
    psi_combined[mid_idx] = 0.5 * (psi_left[mid_idx] + scale * psi_right[mid_idx])
    return psi_combined


def analytical_energy_infinite_well(n: int, d: float) -> float:
    """Return analytical En for a 1D infinite well of width d."""

    return (np.pi**2 * HBAR**2 * n**2) / (2.0 * M_ELECTRON * d**2)


def analytical_wavefunction_infinite_well(x: np.ndarray, n: int, d: float) -> np.ndarray:
    """Return analytical normalized eigenfunction psi_n(x) for centered infinite well."""

    amplitude = np.sqrt(2.0 / d)
    argument = n * np.pi * x / d
    if n % 2 == 1:
        return amplitude * np.cos(argument)
    return amplitude * np.sin(argument)


def scan_brackets(
    x_grid: np.ndarray,
    potential_fn: Potential,
    energy_min: float,
    energy_max: float,
    scan_points: int = 2200,
) -> list[tuple[float, float]]:
    """Find secant start intervals from sign changes in endpoint residual g(E)."""

    energies = np.linspace(energy_min, energy_max, scan_points)
    residuals = np.empty_like(energies)
    for idx, energy in enumerate(energies):
        residuals[idx] = endpoint_residual(float(energy), x_grid, potential_fn)

    brackets: list[tuple[float, float]] = []
    for idx in range(energies.size - 1):
        r1 = residuals[idx]
        r2 = residuals[idx + 1]
        if not (np.isfinite(r1) and np.isfinite(r2)):
            continue
        if r1 == 0.0:
            left = max(energy_min, energies[idx] - (energy_max - energy_min) * 1e-5)
            right = min(energy_max, energies[idx] + (energy_max - energy_min) * 1e-5)
            brackets.append((left, right))
        elif np.signbit(r1) != np.signbit(r2):
            brackets.append((float(energies[idx]), float(energies[idx + 1])))

    return brackets


def solve_state_by_index(
    state_index: int,
    x_grid: np.ndarray,
    potential_fn: Potential,
    energy_window: tuple[float, float],
    energy_tol: float,
    scan_points: int = 1200,
) -> EigenstateResult:
    """Solve for the `state_index`th eigenstate (1-based) in ascending energy."""

    brackets = scan_brackets(
        x_grid=x_grid,
        potential_fn=potential_fn,
        energy_min=energy_window[0],
        energy_max=energy_window[1],
        scan_points=scan_points,
    )

    if len(brackets) < state_index:
        raise RuntimeError(
            f"Only found {len(brackets)} root brackets in {energy_window[0] / E_CHARGE:.3f} to "
            f"{energy_window[1] / E_CHARGE:.3f} eV, but state n={state_index} was requested."
        )

    e1, e2 = brackets[state_index - 1]
    energy, _, iterations, residual = secant_search_eigenvalue(
        energy_1=e1,
        energy_2=e2,
        x_grid=x_grid,
        potential_fn=potential_fn,
        energy_tol=energy_tol,
    )
    psi = matched_wavefunction_from_both_sides(
        energy=energy,
        state_index=state_index,
        x_grid=x_grid,
        potential_fn=potential_fn,
    )
    psi_norm = normalize_wavefunction(psi, x_grid)

    return EigenstateResult(
        state_index=state_index,
        energy_j=energy,
        energy_ev=energy / E_CHARGE,
        iterations=iterations,
        endpoint_residual=residual,
        x=x_grid,
        psi_raw=psi,
        psi_norm=psi_norm,
    )


def solve_state_from_guesses(
    state_index: int,
    x_grid: np.ndarray,
    potential_fn: Potential,
    guess_1: float,
    guess_2: float,
    energy_tol: float,
) -> EigenstateResult:
    """Solve one eigenstate directly from two secant initial energy guesses."""

    energy, _, iterations, residual = secant_search_eigenvalue(
        energy_1=guess_1,
        energy_2=guess_2,
        x_grid=x_grid,
        potential_fn=potential_fn,
        energy_tol=energy_tol,
    )
    psi = matched_wavefunction_from_both_sides(
        energy=energy,
        state_index=state_index,
        x_grid=x_grid,
        potential_fn=potential_fn,
    )
    psi_norm = normalize_wavefunction(psi, x_grid)
    return EigenstateResult(
        state_index=state_index,
        energy_j=energy,
        energy_ev=energy / E_CHARGE,
        iterations=iterations,
        endpoint_residual=residual,
        x=x_grid,
        psi_raw=psi,
        psi_norm=psi_norm,
    )


def solve_lowest_states(
    n_states: int,
    x_grid: np.ndarray,
    potential_fn: Potential,
    energy_window_ev: tuple[float, float],
    energy_tol: float,
    scan_points: int = 1200,
) -> list[EigenstateResult]:
    """Solve and return the lowest `n_states` eigenstates for a potential."""

    energy_window_j = (energy_window_ev[0] * E_CHARGE, energy_window_ev[1] * E_CHARGE)
    brackets = scan_brackets(
        x_grid=x_grid,
        potential_fn=potential_fn,
        energy_min=energy_window_j[0],
        energy_max=energy_window_j[1],
        scan_points=scan_points,
    )
    if len(brackets) < n_states:
        raise RuntimeError(
            f"Only found {len(brackets)} root brackets in {energy_window_ev[0]:.3f} to "
            f"{energy_window_ev[1]:.3f} eV, but {n_states} states were requested."
        )

    results: list[EigenstateResult] = []
    for n in range(1, n_states + 1):
        e1, e2 = brackets[n - 1]
        results.append(
            solve_state_from_guesses(
                state_index=n,
                x_grid=x_grid,
                potential_fn=potential_fn,
                guess_1=e1,
                guess_2=e2,
                energy_tol=energy_tol,
            )
        )
    return results


def save_wavefunction_plot(
    results: Iterable[EigenstateResult],
    a: float,
    title: str,
    outpath: Path,
) -> None:
    """Save normalized wavefunction plot for provided eigenstate results."""

    fig, ax = plt.subplots(figsize=(9, 5))
    for result in results:
        ax.plot(result.x * 1e9, result.psi_norm, lw=2, label=f"n={result.state_index}, E={result.energy_ev:.4f} eV")

    ax.axvline(x=-a * 1e9, color="#5f5f5f", ls="-", lw=2.5)
    ax.axvline(x=+a * 1e9, color="#5f5f5f", ls="-", lw=2.5)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("Normalized psi(x)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_potential_overlay_plot(
    results: Iterable[EigenstateResult],
    potential_fn: Potential,
    x_grid: np.ndarray,
    title: str,
    outpath: Path,
) -> None:
    """Save potential profile with eigenvalues overlaid as horizontal lines."""

    potential_ev = np.asarray(potential_fn(x_grid)) / E_CHARGE
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_grid * 1e9, potential_ev, color="tab:blue", lw=2.5, label="V(x)")

    x_min = float(np.min(x_grid) * 1e9)
    x_max = float(np.max(x_grid) * 1e9)
    for result in results:
        ax.hlines(
            y=result.energy_ev,
            xmin=x_min,
            xmax=x_max,
            colors="tab:red",
            linestyles="--",
            linewidth=1.8,
            label=f"E{result.state_index}" if result.state_index == 1 else None,
        )

    ax.set_xlabel("x (nm)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def nearest_infinite_well_index(energy_j: float, n_max: int = 60) -> int:
    """Return n whose analytical infinite-well energy is closest to `energy_j`."""

    n_values = np.arange(1, n_max + 1)
    energies = analytical_energy_infinite_well(n_values, DOT_SIDE)
    idx = int(np.argmin(np.abs(energies - energy_j)))
    return int(n_values[idx])


def run_infinite_well_section(output_dir: Path) -> dict[str, object]:
    """Run benchmark infinite-well calculations and return summary data."""

    x_grid = make_grid(HALF_WIDTH, DEFAULT_N)
    energy_tol = E_CHARGE / 200_000.0

    def analytic_bracket(n: int, frac: float = 0.08) -> tuple[float, float]:
        exact = analytical_energy_infinite_well(n, DOT_SIDE)
        return exact * (1.0 - frac), exact * (1.0 + frac)

    # Ground and first few excited states (analytical-guided secant seeds)
    well_results = [
        solve_state_from_guesses(
            state_index=n,
            x_grid=x_grid,
            potential_fn=infinite_well_potential,
            guess_1=analytic_bracket(n)[0],
            guess_2=analytic_bracket(n)[1],
            energy_tol=energy_tol,
        )
        for n in (1, 2, 3, 4)
    ]

    # Higher-energy validation states
    high_indices = [16, 24]
    high_states = [
        solve_state_from_guesses(
            state_index=n,
            x_grid=x_grid,
            potential_fn=infinite_well_potential,
            guess_1=analytic_bracket(n)[0],
            guess_2=analytic_bracket(n)[1],
            energy_tol=energy_tol,
        )
        for n in high_indices
    ]

    # Plot ground-state numerical vs analytical
    ground = well_results[0]
    psi_analytic = analytical_wavefunction_infinite_well(x_grid, 1, DOT_SIDE)
    sign_correction = np.sign(trapezoidal_integral(ground.psi_norm * psi_analytic, x_grid))
    psi_num = ground.psi_norm * (1.0 if sign_correction == 0 else sign_correction)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_grid * 1e9, psi_num, lw=2.2, label="Numerical n=1")
    ax.plot(x_grid * 1e9, psi_analytic, lw=1.8, ls="--", label="Analytical n=1")
    ax.axvline(x=-HALF_WIDTH * 1e9, color="#5f5f5f", ls="-", lw=2.5)
    ax.axvline(x=+HALF_WIDTH * 1e9, color="#5f5f5f", ls="-", lw=2.5)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("Normalized psi(x)")
    ax.set_title("Infinite Well: Ground-State Wavefunction")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "infinite_well_ground_comparison.png", dpi=180)
    plt.close(fig)

    save_wavefunction_plot(
        results=well_results,
        a=HALF_WIDTH,
        title="Infinite Well: First Four Normalized Eigenstates",
        outpath=output_dir / "infinite_well_first_four_states.png",
    )

    # Convergence mini-study for n=4
    convergence_rows: list[tuple[int, float, float]] = []
    for n_grid in (1200, 1800, 2400):
        x_local = make_grid(HALF_WIDTH, n_grid)
        e1, e2 = analytic_bracket(4)
        state = solve_state_from_guesses(
            state_index=4,
            x_grid=x_local,
            potential_fn=infinite_well_potential,
            guess_1=e1,
            guess_2=e2,
            energy_tol=energy_tol,
        )
        exact = analytical_energy_infinite_well(4, DOT_SIDE) / E_CHARGE
        abs_err = abs(state.energy_ev - exact)
        convergence_rows.append((n_grid, state.energy_ev, abs_err))

    # Secant stress test: intentionally poor guesses for target n=3
    target_n = 3
    poor_guess_1 = 0.40 * E_CHARGE
    poor_guess_2 = 0.45 * E_CHARGE
    stress_report: dict[str, object] = {"target_n": target_n, "guess_evs": (0.40, 0.45)}
    try:
        stress_energy, _, stress_iter, stress_res = secant_search_eigenvalue(
            energy_1=poor_guess_1,
            energy_2=poor_guess_2,
            x_grid=x_grid,
            potential_fn=infinite_well_potential,
            energy_tol=energy_tol,
            max_iter=60,
        )
        matched_n = nearest_infinite_well_index(stress_energy)
        stress_report.update(
            {
                "outcome": "converged",
                "converged_energy_ev": stress_energy / E_CHARGE,
                "matched_n": matched_n,
                "iterations": stress_iter,
                "residual": stress_res,
                "is_unintended": matched_n != target_n,
            }
        )
    except RuntimeError as exc:
        stress_report.update({"outcome": "failed", "reason": str(exc)})

    return {
        "x_grid": x_grid,
        "well_results": well_results,
        "high_states": high_states,
        "convergence_rows": convergence_rows,
        "stress_report": stress_report,
    }


def run_embedded_potential_sections(output_dir: Path) -> dict[str, object]:
    """Run harmonic-embedded and finite-embedded potential analyses."""

    x_grid = make_grid(HALF_WIDTH, DEFAULT_N)
    energy_tol = E_CHARGE / 200_000.0

    # I. Harmonic embedded potential
    v0_harmonic = 850.0 * E_CHARGE
    harmonic_potential = make_harmonic_embedded_potential(HALF_WIDTH, v0_harmonic)
    harmonic_states = solve_lowest_states(
        n_states=3,
        x_grid=x_grid,
        potential_fn=harmonic_potential,
        energy_window_ev=(0.1, 20.0),
        energy_tol=energy_tol,
    )

    save_wavefunction_plot(
        results=harmonic_states,
        a=HALF_WIDTH,
        title="Embedded Harmonic Potential: Lowest Three Eigenstates",
        outpath=output_dir / "harmonic_embedded_wavefunctions.png",
    )
    save_potential_overlay_plot(
        results=harmonic_states,
        potential_fn=harmonic_potential,
        x_grid=x_grid,
        title="Embedded Harmonic Potential with Lowest Three Eigenvalues",
        outpath=output_dir / "harmonic_embedded_potential_overlay.png",
    )

    omega = np.sqrt(2.0 * v0_harmonic / (M_ELECTRON * HALF_WIDTH**2))
    hbar_omega_ev = (HBAR * omega) / E_CHARGE
    harmonic_spacings = np.diff([state.energy_ev for state in harmonic_states])

    # II. Embedded finite square well
    v0_embedded_square = 500.0 * E_CHARGE
    finite_embedded_potential = make_embedded_finite_square_potential(HALF_WIDTH, v0_embedded_square)
    finite_states = solve_lowest_states(
        n_states=3,
        x_grid=x_grid,
        potential_fn=finite_embedded_potential,
        energy_window_ev=(0.001, 5.0),
        energy_tol=energy_tol,
    )

    save_wavefunction_plot(
        results=finite_states,
        a=HALF_WIDTH,
        title="Embedded Finite Square Well: Lowest Three Eigenstates",
        outpath=output_dir / "finite_embedded_wavefunctions.png",
    )
    save_potential_overlay_plot(
        results=finite_states,
        potential_fn=finite_embedded_potential,
        x_grid=x_grid,
        title="Embedded Finite Square Profile with Lowest Three Eigenvalues",
        outpath=output_dir / "finite_embedded_potential_overlay.png",
    )

    inner_mask = np.abs(x_grid) <= HALF_WIDTH / 2.0
    inner_probabilities = [
        trapezoidal_integral(np.abs(state.psi_norm[inner_mask]) ** 2, x_grid[inner_mask]) for state in finite_states
    ]

    return {
        "harmonic_states": harmonic_states,
        "harmonic_hbar_omega_ev": hbar_omega_ev,
        "harmonic_spacings_ev": harmonic_spacings,
        "finite_states": finite_states,
        "finite_inner_probabilities": inner_probabilities,
    }


def write_summary_report(
    output_dir: Path,
    infinite_data: dict[str, object],
    embedded_data: dict[str, object],
) -> None:
    """Write a compact text summary of numerical outcomes and checks."""

    summary_path = output_dir / "results_summary.txt"

    well_results: list[EigenstateResult] = infinite_data["well_results"]
    high_states: list[EigenstateResult] = infinite_data["high_states"]
    convergence_rows: list[tuple[int, float, float]] = infinite_data["convergence_rows"]
    stress_report: dict[str, object] = infinite_data["stress_report"]

    harmonic_states: list[EigenstateResult] = embedded_data["harmonic_states"]
    finite_states: list[EigenstateResult] = embedded_data["finite_states"]

    lines: list[str] = []
    lines.append("PHAS0029 Numerical Results Summary")
    lines.append("=" * 36)
    lines.append("")

    lines.append("Infinite well: first four states")
    for result in well_results:
        exact_ev = analytical_energy_infinite_well(result.state_index, DOT_SIDE) / E_CHARGE
        rel_err = abs(result.energy_ev - exact_ev) / exact_ev
        lines.append(
            f"n={result.state_index}: E_num={result.energy_ev:.9f} eV, "
            f"E_exact={exact_ev:.9f} eV, rel_err={rel_err:.3e}, "
            f"iter={result.iterations}, |psi(+a)|={result.endpoint_residual:.3e}"
        )

    lines.append("")
    lines.append("High-n checks")
    for result in high_states:
        exact_ev = analytical_energy_infinite_well(result.state_index, DOT_SIDE) / E_CHARGE
        rel_err = abs(result.energy_ev - exact_ev) / exact_ev
        lines.append(
            f"n={result.state_index}: E_num={result.energy_ev:.9f} eV, "
            f"E_exact={exact_ev:.9f} eV, rel_err={rel_err:.3e}"
        )

    lines.append("")
    lines.append("Convergence mini-study (n=4)")
    lines.append("N, E_num(eV), abs_error(eV)")
    for n_grid, energy_ev, abs_err in convergence_rows:
        lines.append(f"{n_grid}, {energy_ev:.9f}, {abs_err:.3e}")

    lines.append("")
    lines.append("Secant stress test")
    if stress_report.get("outcome") == "converged":
        lines.append(
            "poor guesses (0.40,0.45) eV -> "
            f"E={stress_report['converged_energy_ev']:.6f} eV, "
            f"nearest n={stress_report['matched_n']}, "
            f"unintended={stress_report['is_unintended']}"
        )
    else:
        lines.append(f"failed: {stress_report.get('reason')}")

    lines.append("")
    lines.append("Embedded harmonic potential")
    lines.append(f"V0 = 850 eV, hbar*omega estimate = {embedded_data['harmonic_hbar_omega_ev']:.6f} eV")
    for result in harmonic_states:
        lines.append(f"n={result.state_index}: E={result.energy_ev:.6f} eV")
    spacings = embedded_data["harmonic_spacings_ev"]
    lines.append(
        f"spacing(E2-E1,E3-E2)=({spacings[0]:.6f}, {spacings[1]:.6f}) eV"
    )

    lines.append("")
    lines.append("Embedded finite-square potential")
    lines.append("V0 = 500 eV")
    inner_probs = embedded_data["finite_inner_probabilities"]
    for result, inner_prob in zip(finite_states, inner_probs):
        lines.append(f"n={result.state_index}: E={result.energy_ev:.6f} eV, P_inner={inner_prob:.6f}")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Execute all computations and produce figures/report files."""

    output_dir = Path.cwd() / "phas0029_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    infinite_data = run_infinite_well_section(output_dir)
    embedded_data = run_embedded_potential_sections(output_dir)
    write_summary_report(output_dir, infinite_data, embedded_data)

    print(f"Completed. Outputs written to: {output_dir}")
    print(f"Summary report: {output_dir / 'results_summary.txt'}")


if __name__ == "__main__":
    main()
