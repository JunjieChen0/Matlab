"""Control System Toolbox for MatPy."""

import numpy as np
from matpy.builtins import register
from matpy.runtime.types import Mat


class TransferFunction:
    """Transfer function model: G(s) = num(s) / den(s)"""

    def __init__(self, num, den, dt=None):
        self.num = np.atleast_1d(np.array(num, dtype=float))
        self.den = np.atleast_1d(np.array(den, dtype=float))
        self.dt = dt

    def __repr__(self):
        if self.dt is not None:
            return f"Discrete-time transfer function (Ts={self.dt})"
        return "Continuous-time transfer function"

    def __str__(self):
        num_str = self._poly_to_str(self.num)
        den_str = self._poly_to_str(self.den)
        max_len = max(len(num_str), len(den_str))
        return f"  {num_str}\n  {'-' * max_len}\n  {den_str}"

    def _poly_to_str(self, coeffs):
        terms = []
        n = len(coeffs) - 1
        for i, c in enumerate(coeffs):
            if c == 0:
                continue
            power = n - i
            if power == 0:
                terms.append(f"{c:g}")
            elif power == 1:
                terms.append(f"{c:g}s")
            else:
                terms.append(f"{c:g}s^{power}")
        return " + ".join(terms) if terms else "0"

    def poles(self):
        return np.roots(self.den)

    def zeros(self):
        return np.roots(self.num)

    def gain(self):
        return float(np.polyval(self.num, 0) / np.polyval(self.den, 0))

    def evalfr(self, s):
        return np.polyval(self.num, s) / np.polyval(self.den, s)

    def bode_data(self, w=None):
        if w is None:
            w = np.logspace(-2, 2, 500)
        mag = np.zeros(len(w))
        phase = np.zeros(len(w))
        for i, freq in enumerate(w):
            val = self.evalfr(1j * freq)
            mag[i] = 20 * np.log10(np.abs(val))
            phase[i] = np.degrees(np.angle(val))
        return w, mag, phase


class StateSpace:
    """State-space model: x' = Ax + Bu, y = Cx + Du"""

    def __init__(self, A, B, C, D, dt=None):
        self.A = np.atleast_2d(np.array(A, dtype=float))
        self.B = np.atleast_2d(np.array(B, dtype=float))
        self.C = np.atleast_2d(np.array(C, dtype=float))
        self.D = np.atleast_2d(np.array(D, dtype=float))
        self.dt = dt

    def __repr__(self):
        if self.dt is not None:
            return f"Discrete-time state-space model (Ts={self.dt})"
        return "Continuous-time state-space model"

    def poles(self):
        return np.linalg.eigvals(self.A)

    def zeros(self):
        return np.array([])

    def gain(self):
        try:
            return float((self.C @ np.linalg.solve(-self.A, self.B) + self.D).flat[0])
        except np.linalg.LinAlgError:
            return float(self.D.flat[0])


@register("tf")
def _tf(*args):
    """Create transfer function model."""
    if len(args) >= 2:
        num = args[0].data if isinstance(args[0], Mat) else np.array(args[0])
        den = args[1].data if isinstance(args[1], Mat) else np.array(args[1])
        dt = float(args[2]) if len(args) > 2 else None
        return TransferFunction(num.flatten(), den.flatten(), dt)
    return TransferFunction([1], [1, 0])


@register("ss")
def _ss(*args):
    """Create state-space model."""
    if len(args) >= 4:
        A = args[0].data if isinstance(args[0], Mat) else np.array(args[0])
        B = args[1].data if isinstance(args[1], Mat) else np.array(args[1])
        C = args[2].data if isinstance(args[2], Mat) else np.array(args[2])
        D = args[3].data if isinstance(args[3], Mat) else np.array(args[3])
        dt = float(args[4]) if len(args) > 4 else None
        return StateSpace(A, B, C, D, dt)
    return StateSpace([[0]], [[1]], [[1]], [[0]])


@register("zpk")
def _zpk(z, p, k, dt=None):
    """Create zero-pole-gain model."""
    z = z.data if isinstance(z, Mat) else np.array(z).flatten()
    p = p.data if isinstance(p, Mat) else np.array(p).flatten()
    k = float(k.data.flat[0]) if isinstance(k, Mat) else float(k)
    num = k * np.poly(z) if len(z) > 0 else np.array([k])
    den = np.poly(p)
    return TransferFunction(num, den, dt)


@register("pid")
def _pid(Kp, Ki=0, Kd=0):
    """Create PID controller."""
    Kp = float(Kp.data.flat[0]) if isinstance(Kp, Mat) else float(Kp)
    Ki = float(Ki.data.flat[0]) if isinstance(Ki, Mat) else float(Ki)
    Kd = float(Kd.data.flat[0]) if isinstance(Kd, Mat) else float(Kd)
    if Kd != 0:
        num = np.array([Kd, Kp, Ki])
    else:
        num = np.array([Kp, Ki])
    den = np.array([1, 0])
    return TransferFunction(num, den)


@register("feedback")
def _feedback(sys1, sys2=None, sign=-1):
    """Create feedback connection."""
    if not isinstance(sys1, TransferFunction):
        return sys1
    if sys2 is None:
        num_cl = np.polymul(sys1.num, sys1.den)
        den_cl = np.polyadd(sys1.den, sign * sys1.num)
        return TransferFunction(num_cl, den_cl, sys1.dt)
    if not isinstance(sys2, TransferFunction):
        return sys1
    num_ol = np.polymul(sys1.num, sys2.num)
    den_ol = np.polymul(sys1.den, sys2.den)
    num_cl = np.polymul(sys1.num, sys2.den)
    den_cl = np.polyadd(den_ol, sign * num_ol)
    return TransferFunction(num_cl, den_cl, sys1.dt)


@register("series")
def _series(sys1, sys2):
    """Series connection: sys1 * sys2."""
    if not isinstance(sys1, TransferFunction) or not isinstance(sys2, TransferFunction):
        return sys1
    return TransferFunction(np.polymul(sys1.num, sys2.num), np.polymul(sys1.den, sys2.den), sys1.dt)


@register("parallel")
def _parallel(sys1, sys2, sign=1):
    """Parallel connection: sys1 + sign*sys2."""
    if not isinstance(sys1, TransferFunction) or not isinstance(sys2, TransferFunction):
        return sys1
    num1 = np.polymul(sys1.num, sys2.den)
    num2 = np.polymul(sys2.num, sys1.den)
    den = np.polymul(sys1.den, sys2.den)
    return TransferFunction(np.polyadd(num1, sign * num2), den, sys1.dt)


@register("pole")
def _pole(sys):
    if isinstance(sys, (TransferFunction, StateSpace)):
        return Mat(sys.poles())
    return Mat(np.array([]))


@register("zero")
def _zero(sys):
    if isinstance(sys, (TransferFunction, StateSpace)):
        return Mat(sys.zeros())
    return Mat(np.array([]))


@register("dcgain")
def _dcgain(sys):
    if isinstance(sys, (TransferFunction, StateSpace)):
        return sys.gain()
    return 0.0


@register("bode")
def _bode(sys, w=None):
    import matplotlib.pyplot as plt
    if isinstance(sys, TransferFunction):
        w, mag, phase = sys.bode_data(w)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.semilogx(w, mag)
        ax1.set_ylabel("Magnitude (dB)")
        ax1.set_title("Bode Plot")
        ax1.grid(True)
        ax2.semilogx(w, phase)
        ax2.set_ylabel("Phase (deg)")
        ax2.set_xlabel("Frequency (rad/s)")
        ax2.grid(True)
        plt.show()
        return Mat(mag), Mat(phase)
    return Mat(np.array([])), Mat(np.array([]))


@register("step")
def _step(sys, t=None):
    import matplotlib.pyplot as plt
    if isinstance(sys, TransferFunction):
        if t is None:
            poles = sys.poles()
            max_real = np.max(np.abs(np.real(poles))) if len(poles) > 0 else 1
            t_end = 10 / max_real if max_real > 0 else 10
            t = np.linspace(0, t_end, 500)
        else:
            t = t.data if isinstance(t, Mat) else np.array(t)
        y = np.zeros(len(t))
        for i, ti in enumerate(t):
            s_vals = np.linspace(0.01, 50, 100)
            val = np.sum(sys.evalfr(s_vals) * np.exp(s_vals * ti)) / len(s_vals)
            y[i] = np.real(val)
        plt.plot(t, y)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("Step Response")
        plt.grid(True)
        plt.show()
        return Mat(t), Mat(y)
    return Mat(np.array([])), Mat(np.array([]))


@register("nyquist")
def _nyquist(sys, w=None):
    import matplotlib.pyplot as plt
    if isinstance(sys, TransferFunction):
        if w is None:
            w = np.logspace(-2, 2, 500)
        re = np.zeros(len(w))
        im = np.zeros(len(w))
        for i, freq in enumerate(w):
            val = sys.evalfr(1j * freq)
            re[i] = np.real(val)
            im[i] = np.imag(val)
        plt.plot(re, im, re, -im)
        plt.xlabel("Real")
        plt.ylabel("Imaginary")
        plt.title("Nyquist Plot")
        plt.grid(True)
        plt.show()
        return Mat(re), Mat(im)
    return Mat(np.array([])), Mat(np.array([]))


@register("pzmap")
def _pzmap(sys):
    import matplotlib.pyplot as plt
    p = sys.poles() if isinstance(sys, (TransferFunction, StateSpace)) else np.array([])
    z = sys.zeros() if isinstance(sys, (TransferFunction, StateSpace)) else np.array([])
    plt.figure()
    if len(p) > 0:
        plt.plot(np.real(p), np.imag(p), "x", markersize=10, label="Poles")
    if len(z) > 0:
        plt.plot(np.real(z), np.imag(z), "o", markersize=10, label="Zeros")
    plt.axhline(0, color="k", linewidth=0.5)
    plt.axvline(0, color="k", linewidth=0.5)
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title("Pole-Zero Map")
    plt.grid(True)
    plt.legend()
    plt.show()
    return Mat(p), Mat(z)


@register("isstable")
def _isstable(sys):
    if isinstance(sys, (TransferFunction, StateSpace)):
        return bool(np.all(np.real(sys.poles()) < 0))
    return False


@register("issiso")
def _issiso(sys):
    return isinstance(sys, (TransferFunction, StateSpace))


@register("order")
def _order(sys):
    if isinstance(sys, TransferFunction):
        return len(sys.den) - 1
    elif isinstance(sys, StateSpace):
        return sys.A.shape[0]
    return 0


@register("minreal")
def _minreal(sys, tol=1e-6):
    if isinstance(sys, TransferFunction):
        z = sys.zeros()
        p = sys.poles()
        keep_p = np.ones(len(p), dtype=bool)
        keep_z = np.ones(len(z), dtype=bool)
        for i, zi in enumerate(z):
            for j, pj in enumerate(p):
                if keep_p[j] and abs(zi - pj) < tol:
                    keep_p[j] = False
                    keep_z[i] = False
                    break
        new_z = z[keep_z]
        new_p = p[keep_p]
        k = sys.num[0] / sys.den[0]
        num = k * np.poly(new_z) if len(new_z) > 0 else np.array([k])
        den = np.poly(new_p)
        return TransferFunction(num, den, sys.dt)
    return sys


@register("c2d")
def _c2d(sys, Ts, method="zoh"):
    if isinstance(sys, TransferFunction):
        return TransferFunction(sys.num, sys.den, Ts)
    elif isinstance(sys, StateSpace):
        return StateSpace(sys.A, sys.B, sys.C, sys.D, Ts)
    return sys


@register("d2c")
def _d2c(sys, method="zoh"):
    if isinstance(sys, TransferFunction):
        return TransferFunction(sys.num, sys.den, None)
    elif isinstance(sys, StateSpace):
        return StateSpace(sys.A, sys.B, sys.C, sys.D, None)
    return sys


@register("margin")
def _margin(sys):
    if isinstance(sys, TransferFunction):
        w, mag, phase = sys.bode_data()
        gc_idx = np.argmin(np.abs(mag))
        pc_idx = np.argmin(np.abs(phase + 180))
        gm = -mag[pc_idx] if w[pc_idx] > 0 else float("inf")
        pm = 180 + phase[gc_idx] if w[gc_idx] > 0 else float("inf")
        return float(gm), float(pm)
    return 0.0, 0.0


@register("bandwidth")
def _bandwidth(sys):
    if isinstance(sys, TransferFunction):
        w, mag, _ = sys.bode_data()
        idx = np.argmin(np.abs(mag + 3))
        return float(w[idx])
    return 0.0


@register("tfdata")
def _tfdata(sys):
    if isinstance(sys, TransferFunction):
        return Mat(sys.num), Mat(sys.den)
    return Mat(np.array([])), Mat(np.array([]))


@register("ssdata")
def _ssdata(sys):
    if isinstance(sys, StateSpace):
        return Mat(sys.A), Mat(sys.B), Mat(sys.C), Mat(sys.D)
    return Mat(np.array([])), Mat(np.array([])), Mat(np.array([])), Mat(np.array([]))
