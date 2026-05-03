"""tests/unit/test_ifs_arithmetic.py — Unit tests for IFS arithmetic core."""
from __future__ import annotations
import math, pytest
import numpy as np
from src.core.ifs_arithmetic import (
    IFSNumber, _ifs, ifs_add, ifs_multiply, ifs_scalar_multiply, ifs_power,
    ifs_wam, ifs_wgm, vec_wam, vec_wgm, score_function, accuracy_function,
    ifs_compare, hamming_distance, normalized_euclidean_distance,
    score_to_ifs, vec_score_to_ifs, IFSMatrix, ifs_matrix_from_dataframe,
)
from src.core.exceptions import IFSValueError, IFSArithmeticError

A = IFSNumber(0.5, 0.3)   # pi=0.2
B = IFSNumber(0.4, 0.4)   # pi=0.2

class TestIFSNumber:
    def test_pi_derived(self):
        a = IFSNumber(0.5, 0.3)
        assert math.isclose(a.pi, 0.2, abs_tol=1e-9)

    def test_sum_to_one(self):
        a = IFSNumber(0.6, 0.3)
        assert math.isclose(a.mu + a.nu + a.pi, 1.0, abs_tol=1e-9)

    def test_negative_mu_raises(self):
        with pytest.raises(IFSValueError):
            IFSNumber(-0.1, 0.5)

    def test_negative_nu_raises(self):
        with pytest.raises(IFSValueError):
            IFSNumber(0.5, -0.1)

    def test_mu_plus_nu_exceeds_one_raises(self):
        with pytest.raises(IFSValueError):
            IFSNumber(0.7, 0.5)

    def test_boundary_mu_zero(self):
        a = IFSNumber(0.0, 0.8)
        assert a.mu == 0.0 and math.isclose(a.pi, 0.2)

    def test_boundary_nu_zero(self):
        a = IFSNumber(0.8, 0.0)
        assert a.nu == 0.0 and math.isclose(a.pi, 0.2)

    def test_boundary_pi_zero(self):
        a = IFSNumber(0.6, 0.4)
        assert math.isclose(a.pi, 0.0, abs_tol=1e-9)

    def test_fully_member(self):
        a = IFSNumber(1.0, 0.0)
        assert a.mu == 1.0 and a.nu == 0.0 and math.isclose(a.pi, 0.0)

    def test_fully_nonmember(self):
        a = IFSNumber(0.0, 1.0)
        assert a.nu == 1.0 and math.isclose(a.pi, 0.0)

    def test_explicit_pi_consistent(self):
        a = IFSNumber(0.5, 0.3, pi=0.2)
        assert math.isclose(a.pi, 0.2, abs_tol=1e-9)

    def test_explicit_pi_inconsistent_raises(self):
        with pytest.raises(IFSValueError):
            IFSNumber(0.5, 0.3, pi=0.5)

class TestIFSAdd:
    def test_known_result(self):
        c = ifs_add(A, B)
        assert math.isclose(c.mu, 0.5+0.4-0.5*0.4, abs_tol=1e-9)  # 0.7
        assert math.isclose(c.nu, 0.3*0.4, abs_tol=1e-9)            # 0.12
        assert math.isclose(c.pi, 1.0-c.mu-c.nu, abs_tol=1e-9)

    def test_sum_to_one(self):
        c = ifs_add(A, B)
        assert math.isclose(c.mu+c.nu+c.pi, 1.0, abs_tol=1e-9)

    def test_commutativity(self):
        c1 = ifs_add(A, B); c2 = ifs_add(B, A)
        assert math.isclose(c1.mu, c2.mu) and math.isclose(c1.nu, c2.nu)

    def test_identity_with_zero(self):
        zero = IFSNumber(0.0, 1.0)   # pi=0, neutral for addition
        c = ifs_add(A, zero)
        assert math.isclose(c.mu, A.mu, abs_tol=1e-9)

class TestIFSMultiply:
    def test_known_result(self):
        c = ifs_multiply(A, B)
        assert math.isclose(c.mu, 0.5*0.4, abs_tol=1e-9)            # 0.2
        assert math.isclose(c.nu, 0.3+0.4-0.3*0.4, abs_tol=1e-9)   # 0.58

    def test_sum_to_one(self):
        c = ifs_multiply(A, B)
        assert math.isclose(c.mu+c.nu+c.pi, 1.0, abs_tol=1e-9)

    def test_commutativity(self):
        c1 = ifs_multiply(A, B); c2 = ifs_multiply(B, A)
        assert math.isclose(c1.mu, c2.mu) and math.isclose(c1.nu, c2.nu)

class TestIFSScalarMultiply:
    def test_known_result_lambda2(self):
        c = ifs_scalar_multiply(A, 2.0)
        assert math.isclose(c.mu, 1-(1-0.5)**2, abs_tol=1e-9)  # 0.75
        assert math.isclose(c.nu, 0.3**2, abs_tol=1e-9)         # 0.09

    def test_lambda1_identity(self):
        c = ifs_scalar_multiply(A, 1.0)
        assert math.isclose(c.mu, A.mu, abs_tol=1e-9)
        assert math.isclose(c.nu, A.nu, abs_tol=1e-9)

    def test_sum_to_one(self):
        c = ifs_scalar_multiply(A, 3.0)
        assert math.isclose(c.mu+c.nu+c.pi, 1.0, abs_tol=1e-9)

    def test_zero_lambda_raises(self):
        with pytest.raises(IFSArithmeticError):
            ifs_scalar_multiply(A, 0.0)

    def test_negative_lambda_raises(self):
        with pytest.raises(IFSArithmeticError):
            ifs_scalar_multiply(A, -1.0)

class TestIFSPower:
    def test_known_result_power2(self):
        c = ifs_power(A, 2.0)
        assert math.isclose(c.mu, 0.5**2, abs_tol=1e-9)             # 0.25
        assert math.isclose(c.nu, 1-(1-0.3)**2, abs_tol=1e-9)       # 0.51

    def test_power1_identity(self):
        c = ifs_power(A, 1.0)
        assert math.isclose(c.mu, A.mu) and math.isclose(c.nu, A.nu)

    def test_sum_to_one(self):
        c = ifs_power(A, 0.5)
        assert math.isclose(c.mu+c.nu+c.pi, 1.0, abs_tol=1e-9)

    def test_zero_power_raises(self):
        with pytest.raises(IFSArithmeticError):
            ifs_power(A, 0.0)

class TestIFSWAM:
    def test_single_element(self):
        result = ifs_wam([A], [1.0])
        assert math.isclose(result.mu, A.mu, abs_tol=1e-9)
        assert math.isclose(result.nu, A.nu, abs_tol=1e-9)

    def test_known_equal_weights(self):
        # IFWAM([A,B], [0.5,0.5])
        # mu = 1 - (1-0.5)^0.5 * (1-0.4)^0.5 = 1 - sqrt(0.3) ≈ 0.45227
        # nu = 0.3^0.5 * 0.4^0.5 = sqrt(0.12) ≈ 0.34641
        result = ifs_wam([A, B], [0.5, 0.5])
        expected_mu = 1 - math.sqrt(0.5)*math.sqrt(0.6)
        expected_nu = math.sqrt(0.3)*math.sqrt(0.4)
        assert math.isclose(result.mu, expected_mu, abs_tol=1e-6)
        assert math.isclose(result.nu, expected_nu, abs_tol=1e-6)

    def test_weights_auto_normalised(self):
        r1 = ifs_wam([A, B], [0.5, 0.5])
        r2 = ifs_wam([A, B], [1.0, 1.0])
        assert math.isclose(r1.mu, r2.mu, abs_tol=1e-9)

    def test_length_mismatch_raises(self):
        with pytest.raises(IFSArithmeticError):
            ifs_wam([A, B], [0.5])

    def test_zero_weights_raises(self):
        with pytest.raises(IFSArithmeticError):
            ifs_wam([A, B], [0.0, 0.0])

    def test_sum_to_one(self):
        result = ifs_wam([A, B], [0.3, 0.7])
        assert math.isclose(result.mu+result.nu+result.pi, 1.0, abs_tol=1e-9)

class TestIFSWGM:
    def test_single_element(self):
        result = ifs_wgm([A], [1.0])
        assert math.isclose(result.mu, A.mu, abs_tol=1e-9)
        assert math.isclose(result.nu, A.nu, abs_tol=1e-9)

    def test_known_equal_weights(self):
        # IFWGM([A,B], [0.5,0.5])
        # mu = sqrt(0.5*0.4) = sqrt(0.2) ≈ 0.44721
        # nu = 1 - sqrt(0.7*0.6) = 1 - sqrt(0.42) ≈ 0.35191
        result = ifs_wgm([A, B], [0.5, 0.5])
        expected_mu = math.sqrt(0.5*0.4)
        expected_nu = 1 - math.sqrt(0.7*0.6)
        assert math.isclose(result.mu, expected_mu, abs_tol=1e-6)
        assert math.isclose(result.nu, expected_nu, abs_tol=1e-6)

    def test_sum_to_one(self):
        result = ifs_wgm([A, B], [0.3, 0.7])
        assert math.isclose(result.mu+result.nu+result.pi, 1.0, abs_tol=1e-9)

    def test_length_mismatch_raises(self):
        with pytest.raises(IFSArithmeticError):
            ifs_wgm([A, B], [0.5])

class TestScoreFunctions:
    def test_score_formula(self):
        assert math.isclose(score_function(A), 0.5-0.3)  # 0.2

    def test_accuracy_formula(self):
        assert math.isclose(accuracy_function(A), 0.5+0.3)  # 0.8

    def test_score_range_boundary_full_member(self):
        a = IFSNumber(1.0, 0.0)
        assert math.isclose(score_function(a), 1.0)

    def test_score_range_boundary_full_nonmember(self):
        a = IFSNumber(0.0, 1.0)
        assert math.isclose(score_function(a), -1.0)

    def test_score_zero_for_balanced(self):
        a = IFSNumber(0.4, 0.4)
        assert math.isclose(score_function(a), 0.0)

class TestIFSCompare:
    def test_higher_score_wins(self):
        a = IFSNumber(0.7, 0.1)
        b = IFSNumber(0.4, 0.4)
        assert ifs_compare(a, b) == 1
        assert ifs_compare(b, a) == -1

    def test_tiebreak_by_accuracy(self):
        a = IFSNumber(0.5, 0.2)  # S=0.3, H=0.7
        b = IFSNumber(0.4, 0.1)  # S=0.3, H=0.5
        assert ifs_compare(a, b) == 1

    def test_equal(self):
        a = IFSNumber(0.5, 0.3)
        b = IFSNumber(0.5, 0.3)
        assert ifs_compare(a, b) == 0

class TestDistances:
    def test_hamming_known(self):
        d = hamming_distance(A, B)
        # 0.5*(|0.5-0.4|+|0.3-0.4|+|0.2-0.2|) = 0.5*0.2 = 0.1
        assert math.isclose(d, 0.1, abs_tol=1e-9)

    def test_euclidean_known(self):
        d = normalized_euclidean_distance(A, B)
        # sqrt(0.5*(0.01+0.01+0.0)) = sqrt(0.01) = 0.1
        assert math.isclose(d, 0.1, abs_tol=1e-9)

    def test_self_distance_zero(self):
        assert math.isclose(hamming_distance(A, A), 0.0)
        assert math.isclose(normalized_euclidean_distance(A, A), 0.0)

    def test_symmetry(self):
        assert math.isclose(hamming_distance(A, B), hamming_distance(B, A))
        assert math.isclose(
            normalized_euclidean_distance(A, B),
            normalized_euclidean_distance(B, A)
        )

    def test_range_in_01(self):
        a = IFSNumber(1.0, 0.0)
        b = IFSNumber(0.0, 1.0)
        assert 0.0 <= hamming_distance(a, b) <= 1.0
        assert 0.0 <= normalized_euclidean_distance(a, b) <= 1.0

class TestScoreToIFS:
    def test_zero_score(self):
        a = score_to_ifs(0.0, 3.33, pi_fixed=0.05)
        assert math.isclose(a.mu, 0.0, abs_tol=1e-9)
        assert math.isclose(a.pi, 0.05, abs_tol=1e-9)

    def test_max_score(self):
        a = score_to_ifs(3.33, 3.33, pi_fixed=0.05)
        assert math.isclose(a.mu, 0.95, abs_tol=1e-9)
        assert math.isclose(a.nu, 0.0, abs_tol=1e-9)
        assert math.isclose(a.pi, 0.05, abs_tol=1e-9)

    def test_midpoint_score(self):
        a = score_to_ifs(3.33/2, 3.33, pi_fixed=0.05)
        assert math.isclose(a.mu, 0.475, abs_tol=1e-6)
        assert math.isclose(a.nu, 0.475, abs_tol=1e-6)

    def test_sum_to_one(self):
        a = score_to_ifs(2.1, 3.33, pi_fixed=0.05)
        assert math.isclose(a.mu+a.nu+a.pi, 1.0, abs_tol=1e-9)

    def test_invalid_x_max_raises(self):
        with pytest.raises(IFSArithmeticError):
            score_to_ifs(1.0, 0.0)

    def test_invalid_pi_raises(self):
        with pytest.raises(IFSArithmeticError):
            score_to_ifs(1.0, 3.33, pi_fixed=1.5)

    def test_all_benefit_higher_score_higher_mu(self):
        a_low  = score_to_ifs(1.0, 3.33, pi_fixed=0.05)
        a_high = score_to_ifs(2.0, 3.33, pi_fixed=0.05)
        assert a_high.mu > a_low.mu

class TestVecScoreToIFS:
    def test_shape_preserved(self):
        x = np.array([[1.0, 2.0, np.nan], [0.5, 3.33, 1.5]])
        mu, nu, pi = vec_score_to_ifs(x, 3.33, 0.05)
        assert mu.shape == x.shape

    def test_nan_propagated(self):
        x = np.array([np.nan, 1.0])
        mu, nu, pi = vec_score_to_ifs(x, 3.33, 0.05)
        assert np.isnan(mu[0])
        assert not np.isnan(mu[1])

    def test_sum_to_one_non_nan(self):
        x = np.array([0.5, 1.0, 2.0, 3.33])
        mu, nu, pi = vec_score_to_ifs(x, 3.33, 0.05)
        sums = mu + nu + pi
        np.testing.assert_allclose(sums, 1.0, atol=1e-9)

class TestVecWAM:
    def test_single_criterion(self):
        mu = np.array([[0.5], [0.4]])
        nu = np.array([[0.3], [0.4]])
        w  = np.array([1.0])
        ma, na, pa = vec_wam(mu, nu, w)
        np.testing.assert_allclose(ma, [0.5, 0.4], atol=1e-9)
        np.testing.assert_allclose(na, [0.3, 0.4], atol=1e-9)

    def test_sum_to_one(self):
        mu = np.array([[0.5, 0.4], [0.3, 0.6]])
        nu = np.array([[0.3, 0.4], [0.5, 0.2]])
        w  = np.array([0.4, 0.6])
        ma, na, pa = vec_wam(mu, nu, w)
        np.testing.assert_allclose(ma+na+pa, 1.0, atol=1e-9)

    def test_nan_excluded(self):
        mu = np.array([[0.5, np.nan]])
        nu = np.array([[0.3, np.nan]])
        w  = np.array([0.5, 0.5])
        ma, na, pa = vec_wam(mu, nu, w)
        assert math.isclose(ma[0], 0.5, abs_tol=1e-9)

class TestVecWGM:
    def test_sum_to_one(self):
        mu = np.array([[0.5, 0.4], [0.3, 0.6]])
        nu = np.array([[0.3, 0.4], [0.5, 0.2]])
        w  = np.array([0.4, 0.6])
        ma, na, pa = vec_wgm(mu, nu, w)
        np.testing.assert_allclose(ma+na+pa, 1.0, atol=1e-9)

    def test_nan_excluded(self):
        mu = np.array([[0.5, np.nan]])
        nu = np.array([[0.3, np.nan]])
        w  = np.array([0.5, 0.5])
        ma, na, pa = vec_wgm(mu, nu, w)
        assert math.isclose(ma[0], 0.5, abs_tol=1e-9)

class TestIFSMatrixFromDataframe:
    def test_shape(self):
        import pandas as pd
        df = pd.DataFrame(
            {"SC11":[1.0,2.0],"SC12":[0.5,3.0]},
            index=["P01","P02"]
        )
        mat = ifs_matrix_from_dataframe(df, x_max=3.33, pi_fixed=0.05, year=2019)
        assert mat.mu.shape == (2,2)
        assert mat.alternatives == ["P01","P02"]
        assert mat.criteria == ["SC11","SC12"]
        assert mat.year == 2019

    def test_sum_to_one(self):
        import pandas as pd
        df = pd.DataFrame({"SC11":[1.0,2.0],"SC12":[0.5,3.0]}, index=["P01","P02"])
        mat = ifs_matrix_from_dataframe(df, x_max=3.33, pi_fixed=0.05)
        sums = mat.mu + mat.nu + mat.pi
        np.testing.assert_allclose(sums, 1.0, atol=1e-9)

    def test_nan_preserved(self):
        import pandas as pd
        df = pd.DataFrame({"SC11":[np.nan,2.0]}, index=["P01","P02"])
        mat = ifs_matrix_from_dataframe(df, x_max=3.33, pi_fixed=0.05)
        assert np.isnan(mat.mu[0,0])
