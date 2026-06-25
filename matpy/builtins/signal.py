"""Signal Processing Toolbox for MatPy."""

import numpy as np
from scipy import signal as scipy_signal
from matpy.builtins import register
from matpy.runtime.types import Mat


@register("butter")
def _butter(order, Wn, btype="low", analog=False):
    Wn = float(Wn.data.flat[0]) if isinstance(Wn, Mat) else float(Wn)
    b, a = scipy_signal.butter(int(order), Wn, btype=str(btype), analog=bool(analog))
    return Mat(b), Mat(a)


@register("cheby1")
def _cheby1(order, rp, Wn, btype="low", analog=False):
    Wn = float(Wn.data.flat[0]) if isinstance(Wn, Mat) else float(Wn)
    rp = float(rp.data.flat[0]) if isinstance(rp, Mat) else float(rp)
    b, a = scipy_signal.cheby1(int(order), rp, Wn, btype=str(btype), analog=bool(analog))
    return Mat(b), Mat(a)


@register("cheby2")
def _cheby2(order, rs, Wn, btype="low", analog=False):
    Wn = float(Wn.data.flat[0]) if isinstance(Wn, Mat) else float(Wn)
    rs = float(rs.data.flat[0]) if isinstance(rs, Mat) else float(rs)
    b, a = scipy_signal.cheby2(int(order), rs, Wn, btype=str(btype), analog=bool(analog))
    return Mat(b), Mat(a)


@register("ellip")
def _ellip(order, rp, rs, Wn, btype="low", analog=False):
    Wn = float(Wn.data.flat[0]) if isinstance(Wn, Mat) else float(Wn)
    rp = float(rp.data.flat[0]) if isinstance(rp, Mat) else float(rp)
    rs = float(rs.data.flat[0]) if isinstance(rs, Mat) else float(rs)
    b, a = scipy_signal.ellip(int(order), rp, rs, Wn, btype=str(btype), analog=bool(analog))
    return Mat(b), Mat(a)


@register("bessel")
def _bessel(order, Wn, btype="low", analog=False):
    Wn = float(Wn.data.flat[0]) if isinstance(Wn, Mat) else float(Wn)
    b, a = scipy_signal.bessel(int(order), Wn, btype=str(btype), analog=bool(analog))
    return Mat(b), Mat(a)


@register("fir1")
def _fir1(order, Wn, window="hamming"):
    Wn = float(Wn.data.flat[0]) if isinstance(Wn, Mat) else float(Wn)
    b = scipy_signal.firwin(int(order) + 1, Wn, window=str(window))
    return Mat(b)


@register("fir2")
def _fir2(order, f, m, window="hamming"):
    f_data = f.data if isinstance(f, Mat) else np.array(f)
    m_data = m.data if isinstance(m, Mat) else np.array(m)
    b = scipy_signal.firwin2(int(order) + 1, f_data.flatten(), m_data.flatten(), window=str(window))
    return Mat(b)


@register("firls")
def _firls(order, f, a):
    f_data = f.data if isinstance(f, Mat) else np.array(f)
    a_data = a.data if isinstance(a, Mat) else np.array(a)
    b = scipy_signal.firls(int(order) + 1, f_data.flatten(), a_data.flatten())
    return Mat(b)


@register("freqz")
def _freqz(b, a=1, worN=512, whole=False):
    b_data = b.data if isinstance(b, Mat) else np.array(b)
    a_data = a.data if isinstance(a, Mat) else np.array(a) if not isinstance(a, (int, float)) else a
    w, h = scipy_signal.freqz(b_data.flatten(), a_data if isinstance(a, (int, float)) else a_data.flatten(), worN=int(worN), whole=bool(whole))
    return Mat(w), Mat(h)


@register("freqs")
def _freqs(b, a, worN=200):
    b_data = b.data if isinstance(b, Mat) else np.array(b)
    a_data = a.data if isinstance(a, Mat) else np.array(a)
    w, h = scipy_signal.freqs(b_data.flatten(), a_data.flatten(), worN=int(worN))
    return Mat(w), Mat(h)


@register("impz")
def _impz(b, a=1, n=None):
    b_data = b.data if isinstance(b, Mat) else np.array(b)
    a_data = a.data if isinstance(a, Mat) else np.array(a) if not isinstance(a, (int, float)) else a
    if n is None:
        n = 50
    h = scipy_signal.impulse((b_data.flatten(), a_data if isinstance(a, (int, float)) else a_data.flatten()), N=int(n))
    return Mat(h[0]), Mat(h[1])


@register("stepz")
def _stepz(b, a=1, n=None):
    b_data = b.data if isinstance(b, Mat) else np.array(b)
    a_data = a.data if isinstance(a, Mat) else np.array(a) if not isinstance(a, (int, float)) else a
    if n is None:
        n = 50
    h = scipy_signal.step((b_data.flatten(), a_data if isinstance(a, (int, float)) else a_data.flatten()), N=int(n))
    return Mat(h[0]), Mat(h[1])


@register("resample")
def _resample(x, up, down):
    data = x.data if isinstance(x, Mat) else np.array(x)
    result = scipy_signal.resample(data.flatten(), int(len(data) * up / down))
    return Mat(result)


@register("decimate")
def _decimate(x, q, n=None, ftype="iir"):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if n is not None:
        result = scipy_signal.decimate(data.flatten(), int(q), n=int(n), ftype=str(ftype))
    else:
        result = scipy_signal.decimate(data.flatten(), int(q), ftype=str(ftype))
    return Mat(result)


@register("interp")
def _interp(x, r):
    data = x.data if isinstance(x, Mat) else np.array(x)
    result = scipy_signal.resample(data.flatten(), int(len(data) * r))
    return Mat(result)


@register("hilbert")
def _hilbert(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    result = scipy_signal.hilbert(data.flatten())
    return Mat(result)


@register("residuez")
def _residuez(b, a):
    b_data = b.data if isinstance(b, Mat) else np.array(b)
    a_data = a.data if isinstance(a, Mat) else np.array(a)
    r, p, k = scipy_signal.residuez(b_data.flatten(), a_data.flatten())
    return Mat(r), Mat(p), Mat(k)


@register("tf2zpk")
def _tf2zpk(b, a):
    b_data = b.data if isinstance(b, Mat) else np.array(b)
    a_data = a.data if isinstance(a, Mat) else np.array(a)
    z, p, k = scipy_signal.tf2zpk(b_data.flatten(), a_data.flatten())
    return Mat(z), Mat(p), k


@register("zpk2tf")
def _zpk2tf(z, p, k):
    z_data = z.data if isinstance(z, Mat) else np.array(z)
    p_data = p.data if isinstance(p, Mat) else np.array(p)
    k = float(k.data.flat[0]) if isinstance(k, Mat) else float(k)
    b, a = scipy_signal.zpk2tf(z_data.flatten(), p_data.flatten(), k)
    return Mat(b), Mat(a)


@register("sos2tf")
def _sos2tf(sos):
    sos_data = sos.data if isinstance(sos, Mat) else np.array(sos)
    b, a = scipy_signal.sos2tf(sos_data)
    return Mat(b), Mat(a)


@register("tf2sos")
def _tf2sos(b, a):
    b_data = b.data if isinstance(b, Mat) else np.array(b)
    a_data = a.data if isinstance(a, Mat) else np.array(a)
    sos = scipy_signal.tf2sos(b_data.flatten(), a_data.flatten())
    return Mat(sos)


@register("sosfilt")
def _sosfilt(sos, x):
    sos_data = sos.data if isinstance(sos, Mat) else np.array(sos)
    x_data = x.data if isinstance(x, Mat) else np.array(x)
    y = scipy_signal.sosfilt(sos_data, x_data.flatten())
    return Mat(y)


@register("filtfilt")
def _filtfilt(b, a, x):
    b_data = b.data if isinstance(b, Mat) else np.array(b)
    a_data = a.data if isinstance(a, Mat) else np.array(a)
    x_data = x.data if isinstance(x, Mat) else np.array(x)
    y = scipy_signal.filtfilt(b_data.flatten(), a_data.flatten(), x_data.flatten())
    return Mat(y)


@register("lfilter")
def _lfilter(b, a, x, zi=None):
    b_data = b.data if isinstance(b, Mat) else np.array(b)
    a_data = a.data if isinstance(a, Mat) else np.array(a)
    x_data = x.data if isinstance(x, Mat) else np.array(x)
    if zi is not None:
        zi_data = zi.data if isinstance(zi, Mat) else np.array(zi)
        y, zf = scipy_signal.lfilter(b_data.flatten(), a_data.flatten(), x_data.flatten(), zi=zi_data.flatten())
        return Mat(y), Mat(zf)
    y = scipy_signal.lfilter(b_data.flatten(), a_data.flatten(), x_data.flatten())
    return Mat(y)


@register("upfirdn")
def _upfirdn(x, h, up=1, down=1):
    x_data = x.data if isinstance(x, Mat) else np.array(x)
    h_data = h.data if isinstance(h, Mat) else np.array(h)
    y = scipy_signal.upfirdn(h_data.flatten(), x_data.flatten(), int(up), int(down))
    return Mat(y)


@register("fft")
def _fft(x, n=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if n is not None:
        return Mat(np.fft.fft(data.flatten(), int(n)))
    return Mat(np.fft.fft(data.flatten()))


@register("ifft")
def _ifft(x, n=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if n is not None:
        return Mat(np.fft.ifft(data.flatten(), int(n)))
    return Mat(np.fft.ifft(data.flatten()))


@register("fft2")
def _fft2(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.fft.fft2(data))


@register("ifft2")
def _ifft2(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.fft.ifft2(data))


@register("fftshift")
def _fftshift(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.fft.fftshift(data))


@register("ifftshift")
def _ifftshift(x):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(np.fft.ifftshift(data))


@register("fftfreq")
def _fftfreq(n, d=1.0):
    return Mat(np.fft.fftfreq(int(n), float(d)))


@register("periodogram")
def _periodogram(x, fs=1.0, window="hann"):
    data = x.data if isinstance(x, Mat) else np.array(x)
    f, Pxx = scipy_signal.periodogram(data.flatten(), fs=float(fs), window=str(window))
    return Mat(f), Mat(Pxx)


@register("welch")
def _welch(x, fs=1.0, nperseg=256):
    data = x.data if isinstance(x, Mat) else np.array(x)
    f, Pxx = scipy_signal.welch(data.flatten(), fs=float(fs), nperseg=int(nperseg))
    return Mat(f), Mat(Pxx)


@register("spectrogram")
def _spectrogram(x, fs=1.0, nperseg=256):
    data = x.data if isinstance(x, Mat) else np.array(x)
    f, t, Sxx = scipy_signal.spectrogram(data.flatten(), fs=float(fs), nperseg=int(nperseg))
    return Mat(f), Mat(t), Mat(Sxx)


@register("findpeaks")
def _findpeaks(x, height=None, distance=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    kwargs = {}
    if height is not None:
        kwargs["height"] = float(height)
    if distance is not None:
        kwargs["distance"] = int(distance)
    peaks, props = scipy_signal.find_peaks(data.flatten(), **kwargs)
    return Mat(peaks + 1), props


@register("detrend")
def _detrend(x, type="linear"):
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(scipy_signal.detrend(data.flatten(), type=str(type)))


@register("conv")
def _conv(a, b, mode="full"):
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    return Mat(np.convolve(da.flatten(), db.flatten(), mode=str(mode)))


@register("deconv")
def _deconv(a, b):
    da = a.data if isinstance(a, Mat) else np.array(a)
    db = b.data if isinstance(b, Mat) else np.array(b)
    q, r = np.polydiv(da.flatten(), db.flatten())
    return Mat(q), Mat(r)


@register("xcorr")
def _xcorr(x, y=None, maxlags=None):
    xd = x.data if isinstance(x, Mat) else np.array(x)
    yd = y.data if isinstance(y, Mat) else np.array(y) if y is not None else xd
    result = np.correlate(xd.flatten(), yd.flatten(), mode="full")
    lags = np.arange(-len(yd) + 1, len(xd))
    if maxlags is not None:
        center = len(result) // 2
        ml = int(maxlags)
        result = result[center - ml:center + ml + 1]
        lags = lags[center - ml:center + ml + 1]
    return Mat(result), Mat(lags)
