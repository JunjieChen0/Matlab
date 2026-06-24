"""Extra built-in functions for MatPy."""

import numpy as np
from matpy.builtins import register
from matpy.runtime.types import Mat, CellArray, Struct


# ── Image Processing ──────────────────────────────────────────

@register("imread")
def _imread(filename):
    try:
        import matplotlib.image as mpimg
        return Mat(mpimg.imread(str(filename)))
    except Exception as e:
        print(f"Error reading image: {e}")
        return Mat(np.array([]))

@register("imwrite")
def _imwrite(img, filename):
    try:
        import matplotlib.image as mpimg
        data = img.data if isinstance(img, Mat) else np.array(img)
        mpimg.imsave(str(filename), data)
        return 0
    except Exception as e:
        print(f"Error writing image: {e}")
        return -1

@register("imshow")
def _imshow(img):
    import matplotlib.pyplot as plt
    data = img.data if isinstance(img, Mat) else np.array(img)
    plt.imshow(data)
    plt.axis('off')
    plt.show()

@register("rgb2gray")
def _rgb2gray(img):
    data = img.data if isinstance(img, Mat) else np.array(img)
    if data.ndim == 3 and data.shape[2] >= 3:
        return Mat(0.2989 * data[:,:,0] + 0.5870 * data[:,:,1] + 0.1140 * data[:,:,2])
    return img

@register("im2double")
def _im2double(img):
    data = img.data if isinstance(img, Mat) else np.array(img)
    if data.dtype == np.uint8:
        return Mat(data.astype(np.float64) / 255.0)
    return Mat(data.astype(np.float64))

@register("im2uint8")
def _im2uint8(img):
    data = img.data if isinstance(img, Mat) else np.array(img)
    if data.max() <= 1.0:
        return Mat((data * 255).astype(np.uint8))
    return Mat(data.astype(np.uint8))

@register("imresize")
def _imresize(img, scale):
    from scipy.ndimage import zoom
    data = img.data if isinstance(img, Mat) else np.array(img)
    return Mat(zoom(data, float(scale)))

@register("imrotate")
def _imrotate(img, angle):
    from scipy.ndimage import rotate
    data = img.data if isinstance(img, Mat) else np.array(img)
    return Mat(rotate(data, float(angle), reshape=False))

@register("edge")
def _edge(img):
    from scipy.ndimage import sobel
    data = img.data if isinstance(img, Mat) else np.array(img)
    if data.ndim == 3:
        data = np.mean(data, axis=2)
    return Mat(np.hypot(sobel(data, axis=0), sobel(data, axis=1)))


# ── Optimization ──────────────────────────────────────────────

@register("fsolve")
def _fsolve(func, x0, *args):
    from scipy.optimize import fsolve
    x0 = np.array(x0).flatten()
    return Mat(fsolve(func, x0, args=args))

@register("fminunc")
def _fminunc(func, x0, *args):
    from scipy.optimize import minimize
    x0 = np.array(x0).flatten()
    result = minimize(func, x0, args=args, method='BFGS')
    return Mat(result.x), float(result.fun)

@register("lsqnonlin")
def _lsqnonlin(func, x0, *args):
    from scipy.optimize import least_squares
    x0 = np.array(x0).flatten()
    result = least_squares(func, x0, args=args)
    return Mat(result.x), float(result.cost)

@register("lsqcurvefit")
def _lsqcurvefit(func, x0, xdata, ydata, *args):
    from scipy.optimize import curve_fit
    x0 = np.array(x0).flatten()
    xd = xdata.data if isinstance(xdata, Mat) else np.array(xdata)
    yd = ydata.data if isinstance(ydata, Mat) else np.array(ydata)
    popt, pcov = curve_fit(func, xd.flatten(), yd.flatten(), p0=x0)
    return Mat(popt), Mat(pcov)

@register("interp1")
def _interp1_enhanced(x, y, xq, method='linear'):
    from scipy.interpolate import interp1d
    xd = x.data if isinstance(x, Mat) else np.array(x).flatten()
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    xqd = xq.data if isinstance(xq, Mat) else np.array(xq).flatten()
    f = interp1d(xd, yd, kind=str(method), fill_value='extrapolate')
    return Mat(f(xqd))


# ── Statistics ────────────────────────────────────────────────

@register("geomean")
def _geomean(x):
    from scipy.stats import gmean
    data = x.data.flatten() if isinstance(x, Mat) else np.array(x).flatten()
    return float(gmean(data))

@register("harmmean")
def _harmmean(x):
    from scipy.stats import hmean
    data = x.data.flatten() if isinstance(x, Mat) else np.array(x).flatten()
    return float(hmean(data))

@register("trimmean")
def _trimmean(x, percent):
    from scipy.stats import trim_mean
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    return float(trim_mean(data, float(percent) / 100))

@register("iqr")
def _iqr(x):
    from scipy.stats import iqr
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    return float(iqr(data))

@register("kurtosis")
def _kurtosis(x):
    from scipy.stats import kurtosis as scipy_kurt
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    return float(scipy_kurt(data))

@register("skewness")
def _skewness(x):
    from scipy.stats import skew
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    return float(skew(data))

@register("zscore")
def _zscore(x):
    from scipy.stats import zscore as scipy_zscore
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    return Mat(scipy_zscore(data))

@register("normcdf")
def _normcdf(x, mu=0, sigma=1):
    from scipy.stats import norm
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(norm.cdf(data, loc=float(mu), scale=float(sigma)))

@register("norminv")
def _norminv(p, mu=0, sigma=1):
    from scipy.stats import norm
    data = p.data if isinstance(p, Mat) else np.array(p)
    return Mat(norm.ppf(data, loc=float(mu), scale=float(sigma)))

@register("normpdf")
def _normpdf(x, mu=0, sigma=1):
    from scipy.stats import norm
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(norm.pdf(data, loc=float(mu), scale=float(sigma)))

@register("chi2cdf")
def _chi2cdf(x, v):
    from scipy.stats import chi2
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(chi2.cdf(data, df=int(v)))

@register("chi2inv")
def _chi2inv(p, v):
    from scipy.stats import chi2
    data = p.data if isinstance(p, Mat) else np.array(p)
    return Mat(chi2.ppf(data, df=int(v)))

@register("tcdf")
def _tcdf(x, v):
    from scipy.stats import t
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(t.cdf(data, df=int(v)))

@register("tinv")
def _tinv(p, v):
    from scipy.stats import t
    data = p.data if isinstance(p, Mat) else np.array(p)
    return Mat(t.ppf(data, df=int(v)))

@register("fcdf")
def _fcdf(x, v1, v2):
    from scipy.stats import f
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(f.cdf(data, dfn=int(v1), dfd=int(v2)))

@register("finv")
def _finv(p, v1, v2):
    from scipy.stats import f
    data = p.data if isinstance(p, Mat) else np.array(p)
    return Mat(f.ppf(data, dfn=int(v1), dfd=int(v2)))

@register("gamcdf")
def _gamcdf(x, a, b=1):
    from scipy.stats import gamma
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(gamma.cdf(data, a=float(a), scale=float(b)))

@register("gaminv")
def _gaminv(p, a, b=1):
    from scipy.stats import gamma
    data = p.data if isinstance(p, Mat) else np.array(p)
    return Mat(gamma.ppf(data, a=float(a), scale=float(b)))

@register("betacdf")
def _betacdf(x, a, b):
    from scipy.stats import beta
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(beta.cdf(data, a=float(a), b=float(b)))

@register("betainv")
def _betainv(p, a, b):
    from scipy.stats import beta
    data = p.data if isinstance(p, Mat) else np.array(p)
    return Mat(beta.ppf(data, a=float(a), b=float(b)))

@register("unifcdf")
def _unifcdf(x, a, b):
    from scipy.stats import uniform
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(uniform.cdf(data, loc=float(a), scale=float(b) - float(a)))

@register("unifinv")
def _unifinv(p, a, b):
    from scipy.stats import uniform
    data = p.data if isinstance(p, Mat) else np.array(p)
    return Mat(uniform.ppf(data, loc=float(a), scale=float(b) - float(a)))

@register("poisspdf")
def _poisspdf(x, lam):
    from scipy.stats import poisson
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(poisson.pmf(data.astype(int), mu=float(lam)))

@register("poisscdf")
def _poisscdf(x, lam):
    from scipy.stats import poisson
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(poisson.cdf(data.astype(int), mu=float(lam)))

@register("binocdf")
def _binocdf(x, n, p):
    from scipy.stats import binom
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(binom.cdf(data.astype(int), n=int(n), p=float(p)))

@register("binopdf")
def _binopdf(x, n, p):
    from scipy.stats import binom
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(binom.pmf(data.astype(int), n=int(n), p=float(p)))

@register("expcdf")
def _expcdf(x, mu):
    from scipy.stats import expon
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(expon.cdf(data, scale=float(mu)))

@register("exppdf")
def _exppdf(x, mu):
    from scipy.stats import expon
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(expon.pdf(data, scale=float(mu)))

@register("logncdf")
def _logncdf(x, mu, sigma):
    from scipy.stats import lognorm
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(lognorm.cdf(data, s=float(sigma), scale=np.exp(float(mu))))

@register("logninv")
def _logninv(p, mu, sigma):
    from scipy.stats import lognorm
    data = p.data if isinstance(p, Mat) else np.array(p)
    return Mat(lognorm.ppf(data, s=float(sigma), scale=np.exp(float(mu))))

@register("wblcdf")
def _wblcdf(x, a, b):
    from scipy.stats import weibull_min
    data = x.data if isinstance(x, Mat) else np.array(x)
    return Mat(weibull_min.cdf(data, c=float(b), scale=float(a)))

@register("wblinv")
def _wblinv(p, a, b):
    from scipy.stats import weibull_min
    data = p.data if isinstance(p, Mat) else np.array(p)
    return Mat(weibull_min.ppf(data, c=float(b), scale=float(a)))


# ── Hypothesis Testing ────────────────────────────────────────

@register("signrank")
def _signrank(x, y=None):
    from scipy.stats import wilcoxon
    xd = x.data if isinstance(x, Mat) else np.array(x).flatten()
    if y is not None:
        yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
        stat, p = wilcoxon(xd, yd)
    else:
        stat, p = wilcoxon(xd)
    return float(stat), float(p)

@register("ranksum")
def _ranksum(x, y):
    from scipy.stats import ranksums
    xd = x.data if isinstance(x, Mat) else np.array(x).flatten()
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    stat, p = ranksums(xd, yd)
    return float(stat), float(p)

@register("kruskalwallis")
def _kruskalwallis(*args):
    from scipy.stats import kruskal
    groups = []
    for arg in args:
        data = arg.data if isinstance(arg, Mat) else np.array(arg).flatten()
        groups.append(data)
    stat, p = kruskal(*groups)
    return float(stat), float(p)

@register("kstest")
def _kstest(x, name='norm', *args):
    from scipy.stats import kstest as scipy_kstest
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    stat, p = scipy_kstest(data, str(name), args=args)
    return float(stat), float(p)

@register("chi2test")
def _chi2test(observed, expected=None):
    from scipy.stats import chisquare
    obs = observed.data if isinstance(observed, Mat) else np.array(observed).flatten()
    if expected is not None:
        exp = expected.data if isinstance(expected, Mat) else np.array(expected).flatten()
        stat, p = chisquare(obs, exp)
    else:
        stat, p = chisquare(obs)
    return float(stat), float(p)


# ── Resampling ────────────────────────────────────────────────

@register("bootstrp")
def _bootstrp(nboot, func, data):
    d = data.data if isinstance(data, Mat) else np.array(data).flatten()
    results = []
    for _ in range(int(nboot)):
        sample = np.random.choice(d, size=len(d), replace=True)
        results.append(func(sample))
    return Mat(np.array(results))

@register("jackknife")
def _jackknife(func, data):
    d = data.data if isinstance(data, Mat) else np.array(data).flatten()
    results = []
    for i in range(len(d)):
        results.append(func(np.delete(d, i)))
    return Mat(np.array(results))


# ── Linear Models ─────────────────────────────────────────────

@register("fitlm")
def _fitlm(X, y, *args):
    from sklearn.linear_model import LinearRegression
    Xd = X.data if isinstance(X, Mat) else np.array(X)
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    if Xd.ndim == 1:
        Xd = Xd.reshape(-1, 1)
    model = LinearRegression()
    model.fit(Xd, yd)
    return {
        'coefficients': Mat(model.coef_),
        'intercept': float(model.intercept_),
        'r_squared': float(model.score(Xd, yd)),
        'residuals': Mat(yd - model.predict(Xd)),
        'fitted': Mat(model.predict(Xd))
    }

@register("glmfit")
def _glmfit(X, y, distr='normal', *args):
    from sklearn.linear_model import LinearRegression
    Xd = X.data if isinstance(X, Mat) else np.array(X)
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    if Xd.ndim == 1:
        Xd = Xd.reshape(-1, 1)
    model = LinearRegression()
    model.fit(Xd, yd)
    return Mat(model.coef_), float(model.intercept_)


# ── Display ────────────────────────────────────────────────────

@register("tabulate")
def _tabulate(x):
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    unique, counts = np.unique(data, return_counts=True)
    pct = counts / len(data) * 100
    table = np.column_stack([unique, counts, pct])
    print(f"  Value  Count  Percent")
    for row in table:
        print(f"  {int(row[0]):5d}  {int(row[1]):5d}  {row[2]:6.2f}%")
    return Mat(table)

@register("grpstats")
def _grpstats(data, groups, func=None):
    d = data.data if isinstance(data, Mat) else np.array(data).flatten()
    g = groups.data if isinstance(groups, Mat) else np.array(groups).flatten()
    if func is None:
        func = np.mean
    unique_groups = np.unique(g)
    results = []
    for ug in unique_groups:
        mask = g == ug
        results.append(func(d[mask]))
    return Mat(np.array(results))

@register("tiedrank")
def _tiedrank(x):
    from scipy.stats import rankdata
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    return Mat(rankdata(data))
