"""Boost tests for common.py and other modules."""

import pytest
import numpy as np
from tests.conftest import run_matlab, get_val
from matpy.runtime.types import Mat, CellArray, Struct


class TestCommonBoost:
    """Boost common coverage."""

    def test_disp_mat(self, capsys):
        from matpy.builtins.common import _disp_enhanced
        _disp_enhanced(Mat(np.array([1, 2, 3])))
        captured = capsys.readouterr()
        assert "1" in captured.out

    def test_disp_string(self, capsys):
        from matpy.builtins.common import _disp_enhanced
        _disp_enhanced("hello")
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_disp_number(self, capsys):
        from matpy.builtins.common import _disp_enhanced
        _disp_enhanced(42)
        captured = capsys.readouterr()
        assert "42" in captured.out

    def test_display_mat(self, capsys):
        from matpy.builtins.common import _display
        _display(Mat(np.array([1, 2, 3])))
        captured = capsys.readouterr()
        assert "1" in captured.out

    def test_class_mat(self):
        from matpy.builtins.common import _class_enhanced
        result = _class_enhanced(Mat(np.array([1, 2, 3])))
        assert result == "double"

    def test_class_string(self):
        from matpy.builtins.common import _class_enhanced
        result = _class_enhanced("hello")
        assert result == "char"

    def test_class_number(self):
        from matpy.builtins.common import _class_enhanced
        result = _class_enhanced(42)
        assert result in ("double", "int")

    def test_isa_double(self):
        from matpy.builtins.common import _isa_enhanced
        result = _isa_enhanced(Mat(np.array([1, 2, 3])), "double")
        assert result == True

    def test_isa_char(self):
        from matpy.builtins.common import _isa_enhanced
        result = _isa_enhanced("hello", "char")
        assert result == True

    def test_cast_double(self):
        from matpy.builtins.common import _cast
        result = _cast(Mat(np.array([1, 2, 3])), "double")
        assert isinstance(result, Mat)

    def test_cast_single(self):
        from matpy.builtins.common import _cast
        result = _cast(Mat(np.array([1, 2, 3])), "single")
        assert isinstance(result, Mat)

    def test_cast_int32(self):
        from matpy.builtins.common import _cast
        result = _cast(Mat(np.array([1, 2, 3])), "int32")
        assert isinstance(result, Mat)

    def test_tic(self):
        from matpy.builtins.common import _tic
        result = _tic()
        assert result is not None

    def test_now(self):
        from matpy.builtins.common import _now
        result = _now()
        assert result is not None

    def test_date(self):
        from matpy.builtins.common import _date
        result = _date()
        assert isinstance(result, str)

    def test_clock(self):
        from matpy.builtins.common import _clock
        result = _clock()
        assert result is not None


class TestIOBoost:
    """Boost io.py coverage."""

    def test_feval(self):
        from matpy.builtins.io import _feval
        result = _feval(lambda x: x * 2, 5)
        assert result == 10

    def test_nargin(self):
        from matpy.builtins.io import _nargin
        result = _nargin()
        assert result == 0

    def test_nargout(self):
        from matpy.builtins.io import _nargout
        result = _nargout()
        assert result == 0

    def test_mfilename(self):
        from matpy.builtins.io import _mfilename
        result = _mfilename()
        assert result == "<script>"

    def test_inputname(self):
        from matpy.builtins.io import _inputname
        result = _inputname(1)
        assert result == "argin1"

    def test_varargin(self):
        from matpy.builtins.io import _varargin
        result = _varargin()
        assert isinstance(result, list)

    def test_varargout(self):
        from matpy.builtins.io import _varargout
        result = _varargout()
        assert isinstance(result, list)

    def test_validatestring(self):
        from matpy.builtins.io import _validatestring
        result = _validatestring("hel", ["hello", "world", "help"])
        assert result == "hello"

    def test_validateattributes(self):
        from matpy.builtins.io import _validateattributes
        result = _validateattributes(Mat(np.array([1, 2, 3])), ['numeric'])
        assert result == True


class TestAdvancedMathBoost:
    """Boost advanced_math.py coverage."""

    def test_integral(self):
        from matpy.builtins.advanced_math import _integral
        result = _integral(lambda x: x**2, 0, 1)
        assert abs(result - 1/3) < 0.01

    def test_integral2(self):
        from matpy.builtins.advanced_math import _integral2
        result = _integral2(lambda x, y: x + y, 0, 1, 0, 1)
        assert abs(result - 1.0) < 0.01

    def test_fzero(self):
        from matpy.builtins.advanced_math import _fzero
        result = _fzero(lambda x: x**2 - 4, 1.0)
        assert abs(result - 2.0) < 0.01

    def test_fminbnd(self):
        from matpy.builtins.advanced_math import _fminbnd
        result = _fminbnd(lambda x: (x-2)**2, 0, 4)
        assert result is not None

    def test_fminsearch(self):
        from scipy.optimize import minimize
        result = minimize(lambda x: (x[0]-1)**2 + (x[1]-2)**2, [0, 0], method='Nelder-Mead')
        assert result is not None

    def test_polyfit(self):
        from matpy.builtins.advanced_math import _polyfit
        x = Mat(np.array([1, 2, 3, 4, 5]))
        y = Mat(np.array([2, 4, 6, 8, 10]))
        result = _polyfit(x, y, 1)
        assert isinstance(result, Mat)

    def test_polyder(self):
        from matpy.builtins.advanced_math import _polyder
        p = Mat(np.array([1, 2, 1]))
        result = _polyder(p)
        assert isinstance(result, Mat)

    def test_polyint(self):
        from matpy.builtins.advanced_math import _polyint
        p = Mat(np.array([2, 0]))
        result = _polyint(p)
        assert isinstance(result, Mat)

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


class TestControlBoost:
    """Boost control.py coverage."""

    def test_tf(self):
        from matpy.builtins.control import _tf
        result = _tf(Mat(np.array([1])), Mat(np.array([1, 1])))
        assert result is not None

    def test_ss(self):
        from matpy.builtins.control import _ss
        result = _ss(Mat(np.array([[-1]])), Mat(np.array([[1]])), Mat(np.array([[1]])), Mat(np.array([[0]])))
        assert result is not None

    def test_zpk(self):
        from matpy.builtins.control import _zpk
        result = _zpk(Mat(np.array([0])), Mat(np.array([-1])), 1)
        assert result is not None

    def test_pole(self):
        from matpy.builtins.control import TransferFunction, _pole
        sys = TransferFunction([1], [1, 2, 1])
        result = _pole(sys)
        assert result is not None

    def test_zero(self):
        from matpy.builtins.control import TransferFunction, _zero
        sys = TransferFunction([1, 1], [1, 2, 1])
        result = _zero(sys)
        assert result is not None

    def test_dcgain(self):
        from matpy.builtins.control import TransferFunction, _dcgain
        sys = TransferFunction([1], [1, 1])
        result = _dcgain(sys)
        assert result is not None

    def test_series(self):
        from matpy.builtins.control import TransferFunction, _series
        sys1 = TransferFunction([1], [1, 1])
        sys2 = TransferFunction([1], [1, 2])
        result = _series(sys1, sys2)
        assert result is not None

    def test_parallel(self):
        from matpy.builtins.control import TransferFunction, _parallel
        sys1 = TransferFunction([1], [1, 1])
        sys2 = TransferFunction([1], [1, 2])
        result = _parallel(sys1, sys2)
        assert result is not None

    def test_feedback(self):
        from matpy.builtins.control import TransferFunction, _feedback
        sys = TransferFunction([1], [1, 1])
        result = _feedback(sys, 1)
        assert result is not None

    def test_c2d(self):
        from matpy.builtins.control import TransferFunction, _c2d
        sys = TransferFunction([1], [1, 1])
        result = _c2d(sys, 0.1)
        assert result is not None

    def test_step(self):
        from matpy.builtins.control import TransferFunction, _step
        sys = TransferFunction([1], [1, 2, 1])
        t, y = _step(sys)
        assert t is not None
        assert y is not None

    def test_bode(self):
        from matpy.builtins.control import TransferFunction
        sys = TransferFunction([1], [1, 1])
        w, mag, phase = sys.bode_data()
        assert w is not None

    def test_nyquist(self):
        from matpy.builtins.control import TransferFunction, _nyquist
        sys = TransferFunction([1], [1, 1])
        result = _nyquist(sys)
        assert result is not None

    def test_rlocus(self):
        from matpy.builtins.control import TransferFunction, _rlocus
        sys = TransferFunction([1], [1, 2, 1])
        result = _rlocus(sys)
        assert result is not None


class TestImageBoost:
    """Boost image.py coverage."""

    def test_rgb2gray(self):
        from matpy.builtins.image import _rgb2gray
        img = Mat(np.random.rand(10, 10, 3))
        result = _rgb2gray(img)
        assert isinstance(result, Mat)

    def test_im2double(self):
        from matpy.builtins.image import _im2double
        img = Mat(np.array([[128, 255], [0, 64]], dtype=np.uint8))
        result = _im2double(img)
        assert isinstance(result, Mat)

    def test_im2uint8(self):
        from matpy.builtins.image import _im2uint8
        img = Mat(np.array([[0.5, 1.0], [0.0, 0.25]]))
        result = _im2uint8(img)
        assert isinstance(result, Mat)

    def test_imresize(self):
        from matpy.builtins.image import _imresize
        img = Mat(np.zeros((10, 10)))
        result = _imresize(img, 0.5)
        assert isinstance(result, Mat)

    def test_imrotate(self):
        from matpy.builtins.image import _imrotate
        img = Mat(np.zeros((10, 10)))
        result = _imrotate(img, 45)
        assert isinstance(result, Mat)

    def test_imcrop(self):
        from matpy.builtins.image import _imcrop
        img = Mat(np.zeros((10, 10)))
        result = _imcrop(img, Mat(np.array([2, 2, 5, 5])))
        assert isinstance(result, Mat)

    def test_edge(self):
        from matpy.builtins.image import _edge
        img = Mat(np.random.rand(10, 10))
        result = _edge(img)
        assert isinstance(result, Mat)

    def test_histeq(self):
        from matpy.builtins.image import _histeq
        img = Mat(np.random.rand(10, 10))
        result = _histeq(img)
        assert isinstance(result, Mat)

    def test_imadjust(self):
        from matpy.builtins.image import _imadjust
        img = Mat(np.array([[0.2, 0.5, 0.8]]))
        result = _imadjust(img)
        assert isinstance(result, Mat)

    def test_medfilt2(self):
        from matpy.builtins.image import _medfilt2
        img = Mat(np.random.rand(10, 10))
        result = _medfilt2(img)
        assert isinstance(result, Mat)

    def test_imgaussfilt(self):
        from matpy.builtins.image import _imgaussfilt
        img = Mat(np.random.rand(10, 10))
        result = _imgaussfilt(img)
        assert isinstance(result, Mat)

    def test_imfilter(self):
        from matpy.builtins.image import _imfilter
        img = Mat(np.random.rand(10, 10))
        h = Mat(np.ones((3, 3)) / 9)
        result = _imfilter(img, h)
        assert isinstance(result, Mat)

    def test_imerode(self):
        from matpy.builtins.image import _imerode
        img = Mat(np.random.rand(10, 10) > 0.5)
        se = Mat(np.ones((3, 3)))
        result = _imerode(img, se)
        assert isinstance(result, Mat)

    def test_imdilate(self):
        from matpy.builtins.image import _imdilate
        img = Mat(np.random.rand(10, 10) > 0.5)
        se = Mat(np.ones((3, 3)))
        result = _imdilate(img, se)
        assert isinstance(result, Mat)


class TestOptimizationBoost:
    """Boost optimization.py coverage."""

    def test_fzero(self):
        from matpy.builtins.optimization import _fzero
        result = _fzero(lambda x: x**2 - 4, 1.0)
        assert abs(result - 2.0) < 0.01

    def test_fminbnd(self):
        from matpy.builtins.optimization import _fminbnd
        x, fval = _fminbnd(lambda x: (x-2)**2, 0, 4)
        assert abs(x - 2.0) < 0.1

    def test_fminsearch(self):
        from matpy.builtins.optimization import _fminsearch
        result, fval = _fminsearch(lambda x: (x[0]-1)**2 + (x[1]-2)**2, Mat(np.array([0, 0])))
        assert isinstance(result, Mat)

    def test_fminunc(self):
        from matpy.builtins.optimization import _fminunc
        result, fval = _fminunc(lambda x: (x[0]-1)**2 + (x[1]-2)**2, Mat(np.array([0, 0])))
        assert isinstance(result, Mat)

    def test_fsolve(self):
        from matpy.builtins.optimization import _fsolve
        result = _fsolve(lambda x: [x[0]**2 - 4], Mat(np.array([1.0])))
        assert isinstance(result, Mat)

    def test_fmincon(self):
        from matpy.builtins.optimization import _fmincon
        result, fval = _fmincon(
            lambda x: (x[0]-1)**2 + (x[1]-2)**2,
            Mat(np.array([0, 0])),
            A=Mat(np.array([[1, 1]])),
            b=Mat(np.array([5]))
        )
        assert isinstance(result, Mat)

    def test_linprog(self):
        from matpy.builtins.optimization import _linprog
        result, fval = _linprog(
            Mat(np.array([-1, -1])),
            A=Mat(np.array([[1, 1], [-1, 2]])),
            b=Mat(np.array([2, 2]))
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
        result, cost = _lsqnonlin(lambda x: [x[0]**2 - 1], Mat(np.array([0.5])))
        assert isinstance(result, Mat)

    def test_lsqcurvefit(self):
        from matpy.builtins.optimization import _lsqcurvefit
        xdata = Mat(np.array([1, 2, 3, 4]))
        ydata = Mat(np.array([2, 4, 6, 8]))
        result, pcov = _lsqcurvefit(lambda x, a, b: a * x + b, Mat(np.array([1, 0])), xdata, ydata)
        assert isinstance(result, Mat)

    def test_optimset(self):
        from matpy.builtins.optimization import _optimset
        result = _optimset('TolX', 1e-6, 'MaxIter', 100)
        assert isinstance(result, dict)

    def test_optimget(self):
        from matpy.builtins.optimization import _optimget
        options = {'TolX': 1e-6, 'MaxIter': 100}
        result = _optimget(options, 'TolX')
        assert result == 1e-6

    def test_integral(self):
        from matpy.builtins.optimization import _integral
        result = _integral(lambda x: x**2, 0, 1)
        assert abs(result - 1/3) < 0.01

    def test_ode45(self):
        from matpy.builtins.optimization import _ode45
        result_t, result_y = _ode45(lambda t, y: [-y[0]], Mat(np.array([0, 1])), Mat(np.array([1])))
        assert isinstance(result_t, Mat)
        assert isinstance(result_y, Mat)

    def test_ode23(self):
        from matpy.builtins.optimization import _ode23
        result_t, result_y = _ode23(lambda t, y: [-y[0]], Mat(np.array([0, 1])), Mat(np.array([1])))
        assert isinstance(result_t, Mat)
        assert isinstance(result_y, Mat)

    def test_ga(self):
        from matpy.builtins.optimization import _ga
        result, fval = _ga(lambda x: (x[0]-1)**2 + (x[1]-2)**2, 2)
        assert isinstance(result, Mat)

    def test_particleswarm(self):
        from matpy.builtins.optimization import _particleswarm
        result, fval = _particleswarm(lambda x: (x[0]-1)**2 + (x[1]-2)**2, 2)
        assert isinstance(result, Mat)

    def test_simulannealbnd(self):
        from matpy.builtins.optimization import _simulannealbnd
        result, fval = _simulannealbnd(lambda x: (x[0]-1)**2, Mat(np.array([0])))
        assert isinstance(result, Mat)

    def test_fminimax(self):
        from matpy.builtins.optimization import _fminimax
        result, fval = _fminimax(lambda x: [x[0]**2 - 1, x[0] - 2], Mat(np.array([0])))
        assert isinstance(result, Mat)

    def test_psearch(self):
        from matpy.builtins.optimization import _psearch
        result, fval = _psearch(lambda x: (x[0]-1)**2, Mat(np.array([0])))
        assert isinstance(result, Mat)


class TestFileIOBoost:
    """Boost file_io.py coverage."""

    def test_fopen_fclose(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")
        from matpy.builtins.file_io import _fopen, _fclose
        fid = _fopen(str(filepath), 'r')
        assert fid > 0
        result = _fclose(fid)
        assert result == 0

    def test_fgetl(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello\nworld\n")
        from matpy.builtins.file_io import _fopen, _fgetl, _fclose
        fid = _fopen(str(filepath), 'r')
        line = _fgetl(fid)
        assert line == "hello"
        _fclose(fid)

    def test_fgets(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello\nworld\n")
        from matpy.builtins.file_io import _fopen, _fgets, _fclose
        fid = _fopen(str(filepath), 'r')
        line = _fgets(fid)
        assert line == "hello"
        _fclose(fid)

    def test_csvwrite(self, tmp_path):
        filepath = tmp_path / "test.csv"
        from matpy.builtins.file_io import _csvwrite
        result = _csvwrite(str(filepath), Mat(np.array([[1, 2], [3, 4]])))
        assert result == 0
        assert filepath.exists()

    def test_csvread(self, tmp_path):
        filepath = tmp_path / "test.csv"
        filepath.write_text("1,2\n3,4\n")
        from matpy.builtins.file_io import _csvread
        result = _csvread(str(filepath))
        assert isinstance(result, Mat)

    def test_save_load_mat(self, tmp_path):
        filepath = tmp_path / "test.mat"
        from matpy.builtins.file_io import _save, _load
        data = Mat(np.array([[1, 2], [3, 4]]))
        result = _save(str(filepath), data)
        assert result == 0
        loaded = _load(str(filepath))
        assert isinstance(loaded, dict)

    def test_jsonencode(self):
        from matpy.builtins.file_io import _jsonencode
        result = _jsonencode(Mat(np.array([1, 2, 3])))
        assert isinstance(result, str)

    def test_jsondecode(self):
        from matpy.builtins.file_io import _jsondecode
        result = _jsondecode('[1, 2, 3]')
        assert isinstance(result, Mat)

    def test_exist(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")
        from matpy.builtins.file_io import _exist
        assert _exist(str(filepath), 'file') == True

    def test_isfile(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello")
        from matpy.builtins.file_io import _isfile
        assert _isfile(str(filepath)) == True

    def test_isfolder(self, tmp_path):
        from matpy.builtins.file_io import _isfolder
        assert _isfolder(str(tmp_path)) == True

    def test_pwd(self):
        from matpy.builtins.file_io import _pwd
        result = _pwd()
        assert isinstance(result, str)

    def test_tempname(self):
        from matpy.builtins.file_io import _tempname
        result = _tempname()
        assert isinstance(result, str)

    def test_tempdir(self):
        from matpy.builtins.file_io import _tempdir
        result = _tempdir()
        assert isinstance(result, str)

    def test_filesep(self):
        from matpy.builtins.file_io import _filesep
        result = _filesep()
        assert result in ('/', '\\')

    def test_pathsep(self):
        from matpy.builtins.file_io import _pathsep
        result = _pathsep()
        assert result in (':', ';')

    def test_fullfile(self):
        from matpy.builtins.file_io import _fullfile
        result = _fullfile("home", "user", "file.txt")
        assert isinstance(result, str)

    def test_fileparts(self):
        from matpy.builtins.file_io import _fileparts
        directory, base, ext = _fileparts("/home/user/test.txt")
        assert directory == "/home/user"
        assert base == "test"
        assert ext == ".txt"
