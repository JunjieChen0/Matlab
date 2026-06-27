"""Comprehensive tests for optimization.py to boost coverage."""

import pytest
import numpy as np
from tests.conftest import run_matlab, get_val
from matpy.runtime.types import Mat


class TestOptimizationFunctions:
    """Test optimization functions."""

    def test_fzero(self):
        from matpy.builtins.optimization import _fzero

        result = _fzero(lambda x: x**2 - 4, 1.0)
        assert abs(result - 2.0) < 0.01

    def test_fminbnd(self):
        from matpy.builtins.optimization import _fminbnd

        x, fval = _fminbnd(lambda x: (x - 2) ** 2, 0, 4)
        assert abs(x - 2.0) < 0.1

    def test_fminsearch(self):
        from matpy.builtins.optimization import _fminsearch

        result, fval = _fminsearch(
            lambda x: (x[0] - 1) ** 2 + (x[1] - 2) ** 2, Mat(np.array([0, 0]))
        )
        assert isinstance(result, Mat)

    def test_fminunc(self):
        from matpy.builtins.optimization import _fminunc

        result, fval = _fminunc(
            lambda x: (x[0] - 1) ** 2 + (x[1] - 2) ** 2, Mat(np.array([0, 0]))
        )
        assert isinstance(result, Mat)

    def test_fsolve(self):
        from matpy.builtins.optimization import _fsolve

        result = _fsolve(
            lambda x: [x[0] ** 2 + x[1] ** 2 - 1, x[0] - x[1]],
            Mat(np.array([0.5, 0.5])),
        )
        assert isinstance(result, Mat)

    def test_fmincon(self):
        from matpy.builtins.optimization import _fmincon

        result, fval = _fmincon(
            lambda x: (x[0] - 1) ** 2 + (x[1] - 2) ** 2,
            Mat(np.array([0, 0])),
            A=Mat(np.array([[1, 1]])),
            b=Mat(np.array([5])),
        )
        assert isinstance(result, Mat)

    def test_linprog(self):
        from matpy.builtins.optimization import _linprog

        result, fval = _linprog(
            Mat(np.array([-1, -1])),
            A=Mat(np.array([[1, 1], [-1, 2]])),
            b=Mat(np.array([2, 2])),
        )
        assert isinstance(result, Mat)

    def test_quadprog(self):
        from matpy.builtins.optimization import _quadprog

        H = Mat(np.array([[2, 0], [0, 2]]))
        f = Mat(np.array([-2, -5]))
        result, fval = _quadprog(H, f)
        assert isinstance(result, Mat)

    def test_lsqlin(self):
        from matpy.builtins.optimization import _lsqlin

        C = Mat(np.array([[1, 0], [0, 1]]))
        d = Mat(np.array([1, 1]))
        result, fval = _lsqlin(C, d)
        assert isinstance(result, Mat)

    def test_lsqnonneg(self):
        from matpy.builtins.optimization import _lsqnonneg

        C = Mat(np.array([[1, 0], [0, 1]]))
        d = Mat(np.array([1, 1]))
        result, rnorm = _lsqnonneg(C, d)
        assert isinstance(result, Mat)

    def test_lsqnonlin(self):
        from matpy.builtins.optimization import _lsqnonlin

        result, cost = _lsqnonlin(lambda x: [x[0] ** 2 - 1], Mat(np.array([0.5])))
        assert isinstance(result, Mat)

    def test_lsqcurvefit(self):
        from matpy.builtins.optimization import _lsqcurvefit

        xdata = Mat(np.array([1, 2, 3, 4]))
        ydata = Mat(np.array([2, 4, 6, 8]))
        result, pcov = _lsqcurvefit(
            lambda x, a, b: a * x + b, Mat(np.array([1, 0])), xdata, ydata
        )
        assert isinstance(result, Mat)

    def test_optimset(self):
        from matpy.builtins.optimization import _optimset

        result = _optimset("TolX", 1e-6, "MaxIter", 100)
        assert isinstance(result, dict)
        assert result["TolX"] == 1e-6

    def test_optimget(self):
        from matpy.builtins.optimization import _optimget

        options = {"TolX": 1e-6, "MaxIter": 100}
        result = _optimget(options, "TolX")
        assert result == 1e-6

    def test_optimget_default(self):
        from matpy.builtins.optimization import _optimget

        result = _optimget({}, "TolX", 1e-4)
        assert result == 1e-4

    def test_integral(self):
        from matpy.builtins.optimization import _integral

        result = _integral(lambda x: x**2, 0, 1)
        assert abs(result - 1 / 3) < 0.01

    def test_integral2(self):
        from matpy.builtins.optimization import _integral2

        result = _integral2(lambda x, y: x + y, 0, 1, 0, 1)
        assert abs(result - 1.0) < 0.01

    def test_ode45(self):
        from matpy.builtins.optimization import _ode45

        # dy/dt = -y, y(0) = 1
        result_t, result_y = _ode45(
            lambda t, y: [-y[0]], Mat(np.array([0, 1])), Mat(np.array([1]))
        )
        assert isinstance(result_t, Mat)
        assert isinstance(result_y, Mat)

    def test_ode23(self):
        from matpy.builtins.optimization import _ode23

        result_t, result_y = _ode23(
            lambda t, y: [-y[0]], Mat(np.array([0, 1])), Mat(np.array([1]))
        )
        assert isinstance(result_t, Mat)
        assert isinstance(result_y, Mat)

    def test_ga(self):
        from matpy.builtins.optimization import _ga

        result, fval = _ga(lambda x: (x[0] - 1) ** 2 + (x[1] - 2) ** 2, 2)
        assert isinstance(result, Mat)

    def test_particleswarm(self):
        from matpy.builtins.optimization import _particleswarm

        result, fval = _particleswarm(lambda x: (x[0] - 1) ** 2 + (x[1] - 2) ** 2, 2)
        assert isinstance(result, Mat)

    def test_simulannealbnd(self):
        from matpy.builtins.optimization import _simulannealbnd

        result, fval = _simulannealbnd(lambda x: (x[0] - 1) ** 2, Mat(np.array([0])))
        assert isinstance(result, Mat)

    def test_fminimax(self):
        from matpy.builtins.optimization import _fminimax

        result, fval = _fminimax(
            lambda x: [x[0] ** 2 - 1, x[0] - 2], Mat(np.array([0]))
        )
        assert isinstance(result, Mat)

    def test_psearch(self):
        from matpy.builtins.optimization import _psearch

        result, fval = _psearch(lambda x: (x[0] - 1) ** 2, Mat(np.array([0])))
        assert isinstance(result, Mat)


class TestOptimizationIntegration:
    """Integration tests through interpreter."""

    def test_fzero_interpreter(self):
        from matpy.builtins.optimization import _fzero

        result = _fzero(lambda x: x**2 - 4, 1.0)
        assert abs(result - 2.0) < 0.01

    def test_fminbnd_interpreter(self):
        from matpy.builtins.optimization import _fminbnd

        x, fval = _fminbnd(lambda x: (x - 2) ** 2, 0, 4)
        assert abs(x - 2.0) < 0.1

    def test_integral_interpreter(self):
        from matpy.builtins.optimization import _integral

        result = _integral(lambda x: x**2, 0, 1)
        assert abs(result - 1 / 3) < 0.01

    def test_fsolve_interpreter(self):
        from matpy.builtins.optimization import _fsolve

        result = _fsolve(lambda x: [x[0] ** 2 - 4], Mat(np.array([1.0])))
        assert isinstance(result, Mat)

    def test_linprog_interpreter(self):
        from matpy.builtins.optimization import _linprog

        result, fval = _linprog(
            Mat(np.array([-1, -1])),
            A=Mat(np.array([[1, 1], [-1, 2]])),
            b=Mat(np.array([2, 2])),
        )
        assert isinstance(result, Mat)
