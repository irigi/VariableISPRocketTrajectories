import numpy as np
import pytest

from rocketHamilton import AU, MU_SI
from trajectory_scaling import (
    AtlasNormalization,
    compute_canonical_units,
    compute_kappa_tilde,
    compute_rho,
    default_atlas_normalization,
    denormalize_atlas_vector,
    denormalize_solver_seed,
    estimate_kappa_bounds,
    normalize_atlas_vector,
    normalize_solver_seed,
    solve_power_for_kappa,
)


def test_compute_rho_nominal_case():
    rho = compute_rho(1.52 * AU, 1.0 * AU)
    assert np.isclose(rho, 1.52)


def test_compute_rho_rejects_non_positive_radii():
    with pytest.raises(ValueError):
        compute_rho(1.0, 0.0)
    with pytest.raises(ValueError):
        compute_rho(0.0, 1.0)


def test_compute_canonical_units_match_reference_definition():
    units = compute_canonical_units(r0=AU, mu=MU_SI)
    assert np.isclose(units.du_m, AU)
    assert np.isclose(units.tu_s, np.sqrt((AU**3) / MU_SI))


def test_default_atlas_normalization_builds_positive_scale():
    contract = default_atlas_normalization([-9e-5, -22.0, -2800.0, 0.0, -1.5e8])
    scale = contract.scale_array()
    assert scale.shape == (6,)
    assert np.all(scale > 0)


def test_normalization_roundtrip_vector():
    contract = AtlasNormalization(scale=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), offset=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    vector = np.array([1.1, 2.2, -3.0, 4.0, 0.0, 100.0])
    norm = normalize_atlas_vector(vector, contract)
    back = denormalize_atlas_vector(norm, contract)
    np.testing.assert_allclose(back, vector)


def test_normalization_roundtrip_seed_and_time():
    contract = default_atlas_normalization([-9e-5, -22.0, -2800.0, 0.0, -1.5e8], t_scale_days=200.0)
    params = np.array([-9.2e-5, -30.0, -2400.0, 0.01, -1.4e8])
    t_days = 320.0
    norm = normalize_solver_seed(params, t_days, contract)
    params_back, t_back = denormalize_solver_seed(norm, contract)
    np.testing.assert_allclose(params_back, params)
    assert np.isclose(t_back, t_days)


def test_compute_kappa_tilde_matches_formula():
    power = 1.0e9
    m_dry = 1.0e6
    m0 = 3.0e6
    r0 = 1.0 * AU

    expected = (2.0 * power * ((1.0 / m_dry) - (1.0 / m0))) * (r0**2.5) / (MU_SI**1.5)
    got = compute_kappa_tilde(power=power, m_dry=m_dry, m0=m0, r0=r0, mu=MU_SI)

    assert np.isclose(got, expected)


def test_compute_kappa_tilde_monotonic_in_power():
    base = compute_kappa_tilde(power=5.0e8, m_dry=1.0e6, m0=3.0e6, r0=AU, mu=MU_SI)
    stronger = compute_kappa_tilde(power=1.0e9, m_dry=1.0e6, m0=3.0e6, r0=AU, mu=MU_SI)

    assert stronger > base


def test_solve_power_for_kappa_inverts_kappa_formula():
    target_kappa = 8.125
    power = solve_power_for_kappa(target_kappa=target_kappa, m_dry=1.1e6, m0=3.1e6, r0=1.4 * AU, mu=MU_SI)
    kappa = compute_kappa_tilde(power=power, m_dry=1.1e6, m0=3.1e6, r0=1.4 * AU, mu=MU_SI)

    assert np.isclose(kappa, target_kappa)


def test_compute_kappa_tilde_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        compute_kappa_tilde(power=0, m_dry=1.0e6, m0=3.0e6, r0=AU, mu=MU_SI)
    with pytest.raises(ValueError):
        compute_kappa_tilde(power=1.0e9, m_dry=1.0e6, m0=1.0e6, r0=AU, mu=MU_SI)
    with pytest.raises(ValueError):
        compute_kappa_tilde(power=1.0e9, m_dry=1.0e6, m0=3.0e6, r0=-1.0, mu=MU_SI)


def test_estimate_kappa_bounds_expands_range_in_log_space():
    kappa_values = np.array([1.0e-3, 1.0e-1, 1.0e1])
    lo, hi = estimate_kappa_bounds(kappa_values, pad_decades=0.5)

    assert np.isclose(lo, 10 ** (-3.5))
    assert np.isclose(hi, 10 ** (1.5))


def test_estimate_kappa_bounds_rejects_empty_non_positive_or_negative_pad():
    with pytest.raises(ValueError):
        estimate_kappa_bounds(np.array([]))
    with pytest.raises(ValueError):
        estimate_kappa_bounds(np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        estimate_kappa_bounds(np.array([1.0, 2.0]), pad_decades=-0.1)
