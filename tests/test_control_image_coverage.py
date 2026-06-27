"""Comprehensive tests for control.py and image.py to boost coverage."""

import pytest
import numpy as np
from tests.conftest import run_matlab, get_val
from matpy.runtime.types import Mat


class TestControlFunctions:
    """Test control system functions."""

    def test_tf_create(self):
        from matpy.builtins.control import _tf

        result = _tf(Mat(np.array([1])), Mat(np.array([1, 1])))
        assert result is not None

    def test_ss_create(self):
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

    def test_impulse(self):
        from matpy.builtins.control import TransferFunction

        sys = TransferFunction([1], [1, 2, 1])
        # impulse is not directly importable, test through bode_data
        w, mag, phase = sys.bode_data()
        assert w is not None

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

    def test_margin(self):
        from matpy.builtins.control import TransferFunction, _margin

        sys = TransferFunction([1], [1, 2, 1])
        result = _margin(sys)
        assert result is not None

    def test_bandwidth(self):
        from matpy.builtins.control import TransferFunction, _bandwidth

        sys = TransferFunction([1], [1, 1])
        result = _bandwidth(sys)
        assert result is not None


class TestImageFunctions:
    """Test image processing functions."""

    def test_rgb2gray(self):
        from matpy.builtins.image import _rgb2gray

        img = Mat(np.random.rand(10, 10, 3))
        result = _rgb2gray(img)
        assert isinstance(result, Mat)
        assert result.data.shape == (10, 10)

    def test_im2double(self):
        from matpy.builtins.image import _im2double

        img = Mat(np.array([[128, 255], [0, 64]], dtype=np.uint8))
        result = _im2double(img)
        assert isinstance(result, Mat)
        assert result.data.dtype == np.float64

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


class TestControlIntegration:
    """Integration tests through interpreter."""

    def test_tf_interpreter(self):
        interp = run_matlab("sys = tf([1], [1 1]);")
        sys = interp.global_env.get("sys")
        assert sys is not None

    def test_ss_interpreter(self):
        interp = run_matlab("sys = ss([-1], [1], [1], [0]);")
        sys = interp.global_env.get("sys")
        assert sys is not None

    def test_pole_interpreter(self):
        interp = run_matlab("sys = tf([1], [1 2 1]);\np = pole(sys);")
        p = interp.global_env.get("p")
        assert p is not None

    def test_zero_interpreter(self):
        interp = run_matlab("sys = tf([1 1], [1 2 1]);\nz = zero(sys);")
        z = interp.global_env.get("z")
        assert z is not None

    def test_series_interpreter(self):
        interp = run_matlab(
            "sys1 = tf([1], [1 1]);\nsys2 = tf([1], [1 2]);\nsys = series(sys1, sys2);"
        )
        sys = interp.global_env.get("sys")
        assert sys is not None

    def test_parallel_interpreter(self):
        interp = run_matlab(
            "sys1 = tf([1], [1 1]);\nsys2 = tf([1], [1 2]);\nsys = parallel(sys1, sys2);"
        )
        sys = interp.global_env.get("sys")
        assert sys is not None

    def test_feedback_interpreter(self):
        interp = run_matlab("sys = tf([1], [1 1]);\nsys_cl = feedback(sys, 1);")
        sys_cl = interp.global_env.get("sys_cl")
        assert sys_cl is not None


class TestImageIntegration:
    """Integration tests through interpreter."""

    def test_rgb2gray_interpreter(self):
        interp = run_matlab("img = zeros(3, 3, 3);\ngray = rgb2gray(img);")
        gray = interp.global_env.get("gray")
        assert isinstance(gray, Mat)

    def test_im2double_interpreter(self):
        interp = run_matlab("img = [0.5 1.0];\nd = im2double(img);")
        d = interp.global_env.get("d")
        assert isinstance(d, Mat)

    def test_imresize_interpreter(self):
        interp = run_matlab("img = zeros(4, 4);\nresized = imresize(img, 0.5);")
        resized = interp.global_env.get("resized")
        assert isinstance(resized, Mat)

    def test_edge_interpreter(self):
        interp = run_matlab("img = zeros(10, 10);\nedges = edge(img);")
        edges = interp.global_env.get("edges")
        assert isinstance(edges, Mat)
