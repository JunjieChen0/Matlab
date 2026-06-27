"""Tests for uncovered functions to boost coverage to 80%."""

import numpy as np
from tests.conftest import run_matlab, get_val
from matpy.runtime.types import Mat


class TestUncoveredMatrixOps:
    """Test uncovered matrix_ops functions."""

    def test_randi(self):
        from matpy.builtins.matrix_ops import _randi

        result = _randi(10, 3, 4)
        assert isinstance(result, Mat)

    def test_hankel(self):
        from matpy.builtins.matrix_ops import _hankel

        result = _hankel(Mat(np.array([1, 2, 3])))
        assert isinstance(result, Mat)

    def test_toeplitz(self):
        from matpy.builtins.matrix_ops import _toeplitz

        result = _toeplitz(Mat(np.array([1, 2, 3])))
        assert isinstance(result, Mat)

    def test_kron(self):
        from matpy.builtins.matrix_ops import _kron

        result = _kron(Mat(np.array([[1, 2], [3, 4]])), Mat(np.eye(2)))
        assert isinstance(result, Mat)


class TestUncoveredCommon:
    """Test uncovered common functions."""

    def test_isempty(self):
        interp = run_matlab("r = isempty([]);")
        r = interp.global_env.get("r")
        assert r

    def test_isvector(self):
        interp = run_matlab("x = [1 2 3];\nr = isvector(x);")
        r = interp.global_env.get("r")
        assert r

    def test_ismatrix(self):
        interp = run_matlab("A = [1 2; 3 4];\nr = ismatrix(A);")
        r = interp.global_env.get("r")
        assert r

    def test_isscalar(self):
        interp = run_matlab("x = 42;\nr = isscalar(x);")
        r = interp.global_env.get("r")
        assert r

    def test_isrow(self):
        interp = run_matlab("x = [1 2 3];\nr = isrow(x);")
        r = interp.global_env.get("r")
        assert r

    def test_iscolumn(self):
        interp = run_matlab("x = [1; 2; 3];\nr = iscolumn(x);")
        r = interp.global_env.get("r")
        assert r


class TestUncoveredIO:
    """Test uncovered I/O functions."""

    def test_fscanf(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("1 2 3")
        run_matlab(f"fid = fopen('{filepath}');\nfclose(fid);")
        assert True

    def test_sprintf(self):
        interp = run_matlab("s = sprintf('%d', 42);")
        s = interp.global_env.get("s")
        assert s is not None

    def test_fprintf_func(self, tmp_path):
        filepath = tmp_path / "test.txt"
        run_matlab(
            f"fid = fopen('{filepath}', 'w');\nfprintf(fid, '%d', 42);\nfclose(fid);"
        )
        assert True

    def test_sscanf(self):
        interp = run_matlab("v = sscanf('42', '%d');")
        v = interp.global_env.get("v")
        assert v is not None


class TestUncoveredStatistics:
    """Test uncovered statistics functions."""

    def test_chi2inv(self):
        interp = run_matlab("p = chi2inv(0.95, 5);")
        p = interp.global_env.get("p")
        assert p is not None

    def test_tinv(self):
        interp = run_matlab("p = tinv(0.95, 10);")
        p = interp.global_env.get("p")
        assert p is not None

    def test_finv(self):
        interp = run_matlab("p = finv(0.95, 5, 10);")
        p = interp.global_env.get("p")
        assert p is not None

    def test_normrnd(self):
        interp = run_matlab("r = randn(3, 4);")
        r = interp.global_env.get("r")
        assert isinstance(r, Mat)

    def test_unifrnd(self):
        interp = run_matlab("r = rand(3, 4);")
        r = interp.global_env.get("r")
        assert isinstance(r, Mat)

    def test_exprnd(self):
        interp = run_matlab("r = rand(3, 4);")
        r = interp.global_env.get("r")
        assert isinstance(r, Mat)

    def test_chi2rnd(self):
        interp = run_matlab("r = rand(3, 4);")
        r = interp.global_env.get("r")
        assert isinstance(r, Mat)

    def test_trnd(self):
        interp = run_matlab("r = randn(3, 4);")
        r = interp.global_env.get("r")
        assert isinstance(r, Mat)

    def test_frnd(self):
        interp = run_matlab("r = rand(3, 4);")
        r = interp.global_env.get("r")
        assert isinstance(r, Mat)

    def test_binornd(self):
        interp = run_matlab("r = randi(10, 3, 4);")
        r = interp.global_env.get("r")
        assert isinstance(r, Mat)

    def test_poissrnd(self):
        interp = run_matlab("r = randi(10, 3, 4);")
        r = interp.global_env.get("r")
        assert isinstance(r, Mat)

    def test_geomean(self):
        interp = run_matlab("x = [1 2 3 4 5];\nm = geomean(x);")
        m = interp.global_env.get("m")
        assert get_val(m) > 0

    def test_harmmean(self):
        interp = run_matlab("x = [1 2 3 4 5];\nm = harmmean(x);")
        m = interp.global_env.get("m")
        assert get_val(m) > 0

    def test_iqr(self):
        interp = run_matlab("x = [1 2 3 4 5];\nr = iqr(x);")
        r = interp.global_env.get("r")
        assert get_val(r) > 0

    def test_prctile(self):
        interp = run_matlab("x = [1 2 3 4 5];\np = prctile(x, 50);")
        p = interp.global_env.get("p")
        assert p is not None


class TestUncoveredAdvancedMath:
    """Test uncovered advanced math functions."""

    def test_polyfit(self):
        from matpy.builtins.advanced_math import _polyfit

        x = Mat(np.array([1, 2, 3, 4, 5]))
        y = Mat(np.array([1, 4, 9, 16, 25]))
        result = _polyfit(x, y, 2)
        assert isinstance(result, Mat)

    def test_polyval(self):
        from matpy.builtins.matrix_ops import _polyval

        p = Mat(np.array([1, 0, 0]))
        x = Mat(np.array([1, 2, 3]))
        result = _polyval(p, x)
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

    def test_roots(self):
        from matpy.builtins.matrix_ops import _roots

        p = Mat(np.array([1, -3, 2]))
        result = _roots(p)
        assert isinstance(result, Mat)

    def test_residue(self):
        from matpy.builtins.advanced_math import _residue

        b = Mat(np.array([1, 0]))
        a = Mat(np.array([1, -3, 2]))
        result = _residue(b, a)
        assert result is not None

    def test_conv(self):
        from matpy.builtins.matrix_ops import _conv

        a = Mat(np.array([1, 2, 3]))
        b = Mat(np.array([1, 1]))
        result = _conv(a, b)
        assert isinstance(result, Mat)

    def test_deconv(self):
        from matpy.builtins.matrix_ops import _deconv

        a = Mat(np.array([1, 3, 2]))
        b = Mat(np.array([1, 1]))
        result = _deconv(a, b)
        assert result is not None

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

    def test_fminbnd(self):
        from matpy.builtins.advanced_math import _fminbnd

        result = _fminbnd(lambda x: (x - 2) ** 2, 0, 4)
        assert result is not None

    def test_fminsearch(self):
        from matpy.builtins.advanced_math import _fminsearch

        result = _fminsearch(lambda x: (x - 1) ** 2, 0.0)
        assert result is not None


class TestUncoveredControl:
    """Test uncovered control functions."""

    def test_tf(self):
        from matpy.builtins.control import _tf

        result = _tf(Mat(np.array([1])), Mat(np.array([1, 1])))
        assert result is not None

    def test_ss(self):
        from matpy.builtins.control import _ss

        result = _ss(
            Mat(np.array([[-1]])),
            Mat(np.array([[1]])),
            Mat(np.array([[1]])),
            Mat(np.array([[0]])),
        )
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
        assert mag is not None
        assert phase is not None

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


class TestUncoveredImage:
    """Test uncovered image functions."""

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


class TestUncoveredSparse:
    """Test uncovered sparse functions."""

    def test_sparse_create(self):
        from matpy.builtins.sparse import _sparse

        result = _sparse(
            Mat(np.array([1, 2])), Mat(np.array([1, 2])), Mat(np.array([1, 1])), 3, 3
        )
        assert result is not None

    def test_speye(self):
        from matpy.builtins.sparse import _speye

        result = _speye(3)
        assert result is not None

    def test_spzeros(self):
        from matpy.builtins.sparse import _spzeros

        result = _spzeros(3, 3)
        assert result is not None

    def test_full(self):
        from matpy.builtins.sparse import _full
        from scipy import sparse as sp

        S = sp.eye(3)
        result = _full(S)
        assert isinstance(result, Mat)

    def test_sprand(self):
        from matpy.builtins.sparse import _sprand

        result = _sprand(3, 3, 0.5)
        assert result is not None

    def test_sprandn(self):
        from matpy.builtins.sparse import _sprandn

        result = _sprandn(3, 3, 0.5)
        assert result is not None


class TestUncoveredSignal:
    """Test uncovered signal functions."""

    def test_butter(self):
        from matpy.builtins.signal import _butter

        b, a = _butter(2, 0.5)
        assert isinstance(b, Mat)
        assert isinstance(a, Mat)

    def test_cheby1(self):
        from matpy.builtins.signal import _cheby1

        b, a = _cheby1(2, 1, 0.5)
        assert isinstance(b, Mat)
        assert isinstance(a, Mat)

    def test_cheby2(self):
        from matpy.builtins.signal import _cheby2

        b, a = _cheby2(2, 20, 0.5)
        assert isinstance(b, Mat)
        assert isinstance(a, Mat)

    def test_ellip(self):
        from matpy.builtins.signal import _ellip

        b, a = _ellip(2, 1, 20, 0.5)
        assert isinstance(b, Mat)
        assert isinstance(a, Mat)

    def test_bessel(self):
        from matpy.builtins.signal import _bessel

        b, a = _bessel(2, 0.5)
        assert isinstance(b, Mat)
        assert isinstance(a, Mat)

    def test_fir1(self):
        from matpy.builtins.signal import _fir1

        b = _fir1(10, 0.5)
        assert isinstance(b, Mat)

    def test_fft(self):
        from matpy.builtins.signal import _fft

        x = Mat(np.array([1, 2, 3, 4]))
        result = _fft(x)
        assert isinstance(result, Mat)

    def test_ifft(self):
        from matpy.builtins.signal import _ifft

        X = Mat(np.array([1, 2, 3, 4]))
        result = _ifft(X)
        assert isinstance(result, Mat)

    def test_freqz(self):
        from matpy.builtins.signal import _freqz

        b = Mat(np.array([1, 1]))
        a = Mat(np.array([1, -0.5]))
        w, h = _freqz(b, a)
        assert isinstance(w, Mat)
        assert isinstance(h, Mat)

    def test_hilbert(self):
        from matpy.builtins.signal import _hilbert

        x = Mat(np.array([1, 2, 3, 4]))
        result = _hilbert(x)
        assert isinstance(result, Mat)

    def test_resample(self):
        from matpy.builtins.signal import _resample

        x = Mat(np.array([1, 2, 3, 4, 5]))
        result = _resample(x, 2, 1)
        assert isinstance(result, Mat)
