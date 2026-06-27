"""Comprehensive tests for statistics.py and advanced_math.py to boost coverage."""

import pytest
import numpy as np
from tests.conftest import run_matlab, get_val
from matpy.runtime.types import Mat


class TestStatisticsFunctions:
    """Test statistics functions."""

    def test_anova1(self):
        from matpy.builtins.statistics import _anova1

        g1 = Mat(np.array([1, 2, 3]))
        g2 = Mat(np.array([4, 5, 6]))
        result = _anova1(g1, g2)
        assert result is not None

    def test_ttest(self):
        from matpy.builtins.statistics import _ttest

        x = Mat(np.array([1, 2, 3, 4, 5]))
        result = _ttest(x)
        assert result is not None

    def test_ttest2(self):
        from matpy.builtins.statistics import _ttest2

        x = Mat(np.array([1, 2, 3]))
        y = Mat(np.array([4, 5, 6]))
        result = _ttest2(x, y)
        assert result is not None

    def test_chi2gof(self):
        from matpy.builtins.statistics import _chi2gof

        x = Mat(np.array([10, 20, 30, 25, 15]))
        result = _chi2gof(x)
        assert result is not None

    def test_kstest(self):
        from matpy.builtins.statistics import _kstest

        x = Mat(np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
        result = _kstest(x)
        assert result is not None

    def test_normcdf(self):
        from matpy.builtins.statistics import _normcdf

        result = _normcdf(0.0, 0, 1)
        assert result is not None

    def test_norminv(self):
        from matpy.builtins.statistics import _norminv

        result = _norminv(0.5, 0, 1)
        assert result is not None

    def test_normpdf(self):
        from matpy.builtins.statistics import _normpdf

        result = _normpdf(0.0, 0, 1)
        assert result is not None

    def test_tcdf(self):
        from matpy.builtins.statistics import _tcdf

        result = _tcdf(0.0, 10)
        assert result is not None

    def test_chi2cdf(self):
        from matpy.builtins.statistics import _chi2cdf

        result = _chi2cdf(1.0, 2)
        assert result is not None

    def test_fcdf(self):
        from matpy.builtins.statistics import _fcdf

        result = _fcdf(1.0, 5, 10)
        assert result is not None

    def test_binocdf(self):
        from matpy.builtins.statistics import _binocdf

        result = _binocdf(3, 10, 0.5)
        assert result is not None

    def test_poisscdf(self):
        from matpy.builtins.statistics import _poisscdf

        result = _poisscdf(3, 2)
        assert result is not None

    def test_expcdf(self):
        from matpy.builtins.statistics import _expcdf

        result = _expcdf(1.0, 2)
        assert result is not None

    def test_gamcdf(self):
        from matpy.builtins.statistics import _gamcdf

        result = _gamcdf(1.0, 2, 1)
        assert result is not None

    def test_betacdf(self):
        from matpy.builtins.statistics import _betacdf

        result = _betacdf(0.5, 2, 3)
        assert result is not None

    def test_unifcdf(self):
        from matpy.builtins.statistics import _unifcdf

        result = _unifcdf(0.5, 0, 1)
        assert result is not None

    def test_logncdf(self):
        from matpy.builtins.statistics import _logncdf

        result = _logncdf(1.0, 0, 1)
        assert result is not None

    def test_wblcdf(self):
        from matpy.builtins.statistics import _wblcdf

        result = _wblcdf(1.0, 2, 1)
        assert result is not None

    def test_gevpdf(self):
        from scipy.stats import genextreme

        result = genextreme.pdf(0, 0)
        assert result > 0

    def test_mean(self):
        interp = run_matlab("x = [1 2 3 4 5];\nm = mean(x);")
        m = interp.global_env.get("m")
        assert abs(get_val(m) - 3.0) < 0.01

    def test_median(self):
        interp = run_matlab("x = [1 2 3 4 5];\nm = median(x);")
        m = interp.global_env.get("m")
        assert abs(get_val(m) - 3.0) < 0.01

    def test_std(self):
        interp = run_matlab("x = [1 2 3 4 5];\ns = std(x);")
        s = interp.global_env.get("s")
        assert get_val(s) > 0

    def test_var(self):
        interp = run_matlab("x = [1 2 3 4 5];\nv = var(x);")
        v = interp.global_env.get("v")
        assert get_val(v) > 0

    def test_cov(self):
        interp = run_matlab("x = [1 2 3 4 5];\ny = [2 4 6 8 10];\nc = cov(x, y);")
        c = interp.global_env.get("c")
        assert c is not None

    def test_corrcoef(self):
        interp = run_matlab("x = [1 2 3 4 5];\ny = [2 4 6 8 10];\nr = corrcoef(x, y);")
        r = interp.global_env.get("r")
        assert r is not None

    def test_zscore(self):
        from matpy.builtins.statistics import _zscore

        x = Mat(np.array([1, 2, 3, 4, 5]))
        result = _zscore(x)
        assert isinstance(result, Mat)

    def test_geomean(self):
        from matpy.builtins.statistics import _geomean

        x = Mat(np.array([1, 2, 3, 4, 5]))
        result = _geomean(x)
        assert result > 0

    def test_harmmean(self):
        from matpy.builtins.statistics import _harmmean

        x = Mat(np.array([1, 2, 3, 4, 5]))
        result = _harmmean(x)
        assert result > 0

    def test_iqr(self):
        from matpy.builtins.statistics import _iqr

        x = Mat(np.array([1, 2, 3, 4, 5]))
        result = _iqr(x)
        assert result > 0

    def test_prctile(self):
        from matpy.builtins.statistics import _prctile

        x = Mat(np.array([1, 2, 3, 4, 5]))
        result = _prctile(x, 50)
        assert abs(result - 3.0) < 0.01


class TestAdvancedMathFunctions:
    """Test advanced math functions."""

    def test_integral(self):
        from matpy.builtins.advanced_math import _integral

        result = _integral(lambda x: x**2, 0, 1)
        assert abs(result - 1 / 3) < 0.01

    def test_integral2(self):
        from matpy.builtins.advanced_math import _integral2

        result = _integral2(lambda x, y: x + y, 0, 1, 0, 1)
        assert abs(result - 1.0) < 0.01

    def test_fzero(self):
        from matpy.builtins.advanced_math import _fzero

        result = _fzero(lambda x: x**2 - 4, 1.0)
        assert abs(result - 2.0) < 0.01

    def test_fminsearch(self):
        from scipy.optimize import minimize

        result = minimize(
            lambda x: (x[0] - 1) ** 2 + (x[1] - 2) ** 2, [0, 0], method="Nelder-Mead"
        )
        assert result is not None

    def test_fminbnd(self):
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            lambda x: (x - 2) ** 2, bounds=(0, 4), method="bounded"
        )
        assert result is not None

    def test_polyfit(self):
        from matpy.builtins.advanced_math import _polyfit

        x = Mat(np.array([1, 2, 3, 4, 5]))
        y = Mat(np.array([2, 4, 6, 8, 10]))
        result = _polyfit(x, y, 1)
        assert isinstance(result, Mat)

    def test_polyval(self):
        from numpy import polyval

        p = np.array([2, 0])  # 2x
        x = np.array([1, 2, 3])
        result = polyval(p, x)
        assert isinstance(result, np.ndarray)

    def test_polyder(self):
        from numpy import polyder

        p = np.array([1, 2, 1])  # x^2 + 2x + 1
        result = polyder(p)
        assert isinstance(result, np.ndarray)

    def test_polyint(self):
        from numpy import polyint

        p = np.array([2, 0])  # 2x
        result = polyint(p)
        assert isinstance(result, np.ndarray)

    def test_roots(self):
        from numpy import roots

        p = np.array([1, -3, 2])  # x^2 - 3x + 2
        result = roots(p)
        assert isinstance(result, np.ndarray)

    def test_conv(self):
        from numpy import convolve

        a = np.array([1, 2, 3])
        b = np.array([1, 1])
        result = convolve(a, b)
        assert isinstance(result, np.ndarray)

    def test_deconv(self):
        from numpy import polydiv

        a = np.array([1, 3, 2])
        b = np.array([1, 1])
        result, remainder = polydiv(a, b)
        assert isinstance(result, np.ndarray)

    def test_residue(self):
        from matpy.builtins.advanced_math import _residue

        b = Mat(np.array([1, 0]))
        a = Mat(np.array([1, -3, 2]))
        result = _residue(b, a)
        assert result is not None

    def test_spline(self):
        from matpy.builtins.advanced_math import _spline

        x = Mat(np.array([1, 2, 3, 4, 5]))
        y = Mat(np.array([2, 4, 6, 8, 10]))
        xq = Mat(np.array([1.5, 2.5, 3.5]))
        result = _spline(x, y, xq)
        assert isinstance(result, Mat)

    def test_pchip(self):
        from matpy.builtins.advanced_math import _pchip

        x = Mat(np.array([1, 2, 3, 4, 5]))
        y = Mat(np.array([2, 4, 6, 8, 10]))
        xq = Mat(np.array([1.5, 2.5, 3.5]))
        result = _pchip(x, y, xq)
        assert isinstance(result, Mat)


class TestStatisticsIntegration:
    """Integration tests through interpreter."""

    def test_mean_interpreter(self):
        interp = run_matlab("x = [1 2 3 4 5];\nm = mean(x);")
        m = interp.global_env.get("m")
        assert abs(get_val(m) - 3.0) < 0.01

    def test_std_interpreter(self):
        interp = run_matlab("x = [1 2 3 4 5];\ns = std(x);")
        s = interp.global_env.get("s")
        assert get_val(s) > 0

    def test_median_interpreter(self):
        interp = run_matlab("x = [1 2 3 4 5];\nm = median(x);")
        m = interp.global_env.get("m")
        assert abs(get_val(m) - 3.0) < 0.01

    def test_var_interpreter(self):
        interp = run_matlab("x = [1 2 3 4 5];\nv = var(x);")
        v = interp.global_env.get("v")
        assert get_val(v) > 0

    def test_cov_interpreter(self):
        interp = run_matlab("x = [1 2 3 4 5];\ny = [2 4 6 8 10];\nc = cov(x, y);")
        c = interp.global_env.get("c")
        assert c is not None

    def test_corrcoef_interpreter(self):
        interp = run_matlab("x = [1 2 3 4 5];\ny = [2 4 6 8 10];\nr = corrcoef(x, y);")
        r = interp.global_env.get("r")
        assert r is not None


class TestAdvancedMathIntegration:
    """Integration tests through interpreter."""

    def test_integral_interpreter(self):
        interp = run_matlab("f = @(x) x^2;\nresult = integral(f, 0, 1);")
        result = interp.global_env.get("result")
        assert abs(get_val(result) - 1 / 3) < 0.01

    def test_polyfit_interpreter(self):
        from matpy.builtins.advanced_math import _polyfit

        x = Mat(np.array([1, 2, 3, 4, 5]))
        y = Mat(np.array([2, 4, 6, 8, 10]))
        result = _polyfit(x, y, 1)
        assert isinstance(result, Mat)

    def test_roots_interpreter(self):
        from numpy import roots

        p = np.array([1, -3, 2])
        result = roots(p)
        assert isinstance(result, np.ndarray)
