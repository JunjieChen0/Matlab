"""Statistics Toolbox for MatPy."""

import numpy as np
from scipy import stats as scipy_stats
from matpy.builtins import register
from matpy.runtime.types import Mat


@register("fitlm")
def _fitlm(X, y, *args):
    from sklearn.linear_model import LinearRegression
    Xd = X.data if isinstance(X, Mat) else np.array(X)
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    if Xd.ndim == 1:
        Xd = Xd.reshape(-1, 1)
    model = LinearRegression()
    model.fit(Xd, yd)
    y_pred = model.predict(Xd)
    residuals = yd - y_pred
    r_squared = model.score(Xd, yd)
    return {
        "coefficients": Mat(model.coef_),
        "intercept": float(model.intercept_),
        "r_squared": float(r_squared),
        "residuals": Mat(residuals),
        "fitted": Mat(y_pred),
    }


@register("stepwiselm")
def _stepwiselm(X, y, *args):
    from sklearn.linear_model import LinearRegression
    Xd = X.data if isinstance(X, Mat) else np.array(X)
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    if Xd.ndim == 1:
        Xd = Xd.reshape(-1, 1)
    model = LinearRegression()
    model.fit(Xd, yd)
    return Mat(model.coef_), float(model.intercept_), float(model.score(Xd, yd))


@register("glmfit")
def _glmfit(X, y, distr="normal", *args):
    from sklearn.linear_model import LinearRegression
    Xd = X.data if isinstance(X, Mat) else np.array(X)
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    if Xd.ndim == 1:
        Xd = Xd.reshape(-1, 1)
    model = LinearRegression()
    model.fit(Xd, yd)
    return Mat(model.coef_), float(model.intercept_)


@register("anova1")
def _anova1(*args):
    groups = []
    for arg in args:
        data = arg.data if isinstance(arg, Mat) else np.array(arg).flatten()
        groups.append(data)
    f_stat, p_value = scipy_stats.f_oneway(*groups)
    return float(f_stat), float(p_value)


@register("anova2")
def _anova2(*args):
    groups = []
    for arg in args:
        data = arg.data if isinstance(arg, Mat) else np.array(arg).flatten()
        groups.append(data)
    f_stat, p_value = scipy_stats.f_oneway(*groups)
    return float(f_stat), float(p_value)


@register("anovan")
def _anovan(y, groups, *args):
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    if isinstance(groups, Mat):
        groups = groups.data
    unique_groups = np.unique(groups)
    group_data = [yd[groups == g] for g in unique_groups]
    f_stat, p_value = scipy_stats.f_oneway(*group_data)
    return float(f_stat), float(p_value)


@register("ttest")
def _ttest(x, mu=0):
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    mu = float(mu.data.flat[0]) if isinstance(mu, Mat) else float(mu)
    t_stat, p_value = scipy_stats.ttest_1samp(data, mu)
    return float(t_stat), float(p_value)


@register("ttest2")
def _ttest2(x, y):
    xd = x.data if isinstance(x, Mat) else np.array(x).flatten()
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    t_stat, p_value = scipy_stats.ttest_ind(xd, yd)
    return float(t_stat), float(p_value)


@register("ttest_paired")
def _ttest_paired(x, y):
    xd = x.data if isinstance(x, Mat) else np.array(x).flatten()
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    t_stat, p_value = scipy_stats.ttest_rel(xd, yd)
    return float(t_stat), float(p_value)


@register("signrank")
def _signrank(x, y=None):
    xd = x.data if isinstance(x, Mat) else np.array(x).flatten()
    if y is not None:
        yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
        stat, p = scipy_stats.wilcoxon(xd, yd)
    else:
        stat, p = scipy_stats.wilcoxon(xd)
    return float(stat), float(p)


@register("ranksum")
def _ranksum(x, y):
    xd = x.data if isinstance(x, Mat) else np.array(x).flatten()
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    stat, p = scipy_stats.ranksums(xd, yd)
    return float(stat), float(p)


@register("kruskalwallis")
def _kruskalwallis(*args):
    groups = []
    for arg in args:
        data = arg.data if isinstance(arg, Mat) else np.array(arg).flatten()
        groups.append(data)
    stat, p = scipy_stats.kruskal(*groups)
    return float(stat), float(p)


@register("friedman")
def _friedman(*args):
    groups = []
    for arg in args:
        data = arg.data if isinstance(arg, Mat) else np.array(arg).flatten()
        groups.append(data)
    stat, p = scipy_stats.friedmanchisquare(*groups)
    return float(stat), float(p)


@register("chi2gof")
def _chi2gof(x, nbins=10):
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    observed, bins = np.histogram(data, bins=int(nbins))
    expected = np.ones_like(observed) * len(data) / int(nbins)
    chi2, p = scipy_stats.chisquare(observed, expected)
    return float(chi2), float(p)


@register("kstest")
def _kstest(x, name="norm", *args):
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    stat, p = scipy_stats.kstest(data, str(name), args=args)
    return float(stat), float(p)


@register("lillietest")
def _lillietest(x):
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    stat, p = scipy_stats.lilliefors(data)
    return float(stat), float(p)


@register("adtest")
def _adtest(x):
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    result = scipy_stats.anderson(data)
    return float(result.statistic), Mat(result.critical_values)


@register("vartest")
def _vartest(x, v):
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    v = float(v.data.flat[0]) if isinstance(v, Mat) else float(v)
    n = len(data)
    s2 = np.var(data, ddof=1)
    chi2_stat = (n - 1) * s2 / v
    p = 1 - scipy_stats.chi2.cdf(chi2_stat, n - 1)
    return float(chi2_stat), float(p)


@register("vartest2")
def _vartest2(x, y):
    xd = x.data if isinstance(x, Mat) else np.array(x).flatten()
    yd = y.data if isinstance(y, Mat) else np.array(y).flatten()
    f_stat = np.var(xd, ddof=1) / np.var(yd, ddof=1)
    p = 1 - scipy_stats.f.cdf(f_stat, len(xd) - 1, len(yd) - 1)
    return float(f_stat), float(p)


@register("chi2test")
def _chi2test(observed, expected=None):
    obs = observed.data if isinstance(observed, Mat) else np.array(observed).flatten()
    if expected is not None:
        exp = expected.data if isinstance(expected, Mat) else np.array(expected).flatten()
        stat, p = scipy_stats.chisquare(obs, exp)
    else:
        stat, p = scipy_stats.chisquare(obs)
    return float(stat), float(p)


@register("pca")
def _pca(x, *args):
    from sklearn.decomposition import PCA
    data = x.data if isinstance(x, Mat) else np.array(x)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_components = int(args[0]) if args else min(data.shape)
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return Mat(pca.components_), Mat(pca.explained_variance_ratio_), Mat(pca.transform(data))


@register("factoran")
def _factoran(x, n_factors=1):
    from sklearn.decomposition import FactorAnalysis
    data = x.data if isinstance(x, Mat) else np.array(x)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    fa = FactorAnalysis(n_components=int(n_factors))
    fa.fit(data)
    return Mat(fa.components_), Mat(fa.noise_variance_)


@register("kmeans")
def _kmeans(x, k, *args):
    from scipy.cluster.vq import kmeans as scipy_kmeans
    data = x.data if isinstance(x, Mat) else np.array(x)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    centroids, distortion = scipy_kmeans(data, int(k))
    return Mat(centroids), float(distortion)


@register("linkage")
def _linkage(x, method="ward"):
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    data = x.data if isinstance(x, Mat) else np.array(x)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    Z = scipy_linkage(data, method=str(method))
    return Mat(Z)


@register("cluster")
def _cluster(Z, maxclust=2):
    from scipy.cluster.hierarchy import fcluster
    Z_data = Z.data if isinstance(Z, Mat) else np.array(Z)
    T = fcluster(Z_data, int(maxclust), criterion="maxclust")
    return Mat(T)


@register("dendrogram")
def _dendrogram(Z):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
    Z_data = Z.data if isinstance(Z, Mat) else np.array(Z)
    scipy_dendrogram(Z_data)
    plt.title("Dendrogram")
    plt.show()


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


@register("mle")
def _mle(data, name="normal"):
    d = data.data if isinstance(data, Mat) else np.array(data).flatten()
    name = str(name).lower()
    if name == "normal":
        mu, sigma = scipy_stats.norm.fit(d)
        return Mat(np.array([mu, sigma]))
    elif name == "exponential":
        mu = scipy_stats.expon.fit(d)[1]
        return Mat(np.array([mu]))
    elif name == "weibull":
        c, loc, scale = scipy_stats.weibull_min.fit(d)
        return Mat(np.array([scale, c]))
    return Mat(np.array([]))


@register("zscore")
def _zscore(x):
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    return Mat(scipy_stats.zscore(data))


@register("tiedrank")
def _tiedrank(x):
    data = x.data if isinstance(x, Mat) else np.array(x).flatten()
    return Mat(scipy_stats.rankdata(data))


@register("geomean")
def _geomean(x):
    data = x.data.flatten() if isinstance(x, Mat) else np.array(x).flatten()
    return float(scipy_stats.gmean(data))


@register("harmmean")
def _harmmean(x):
    data = x.data.flatten() if isinstance(x, Mat) else np.array(x).flatten()
    return float(scipy_stats.hmean(data))


@register("trimmean")
def _trimmean(x, percent):
    data = x.data.flatten() if isinstance(x, Mat) else np.array(x).flatten()
    percent = float(percent.data.flat[0]) if isinstance(percent, Mat) else float(percent)
    return float(scipy_stats.trim_mean(data, percent / 100))


@register("iqr")
def _iqr(x):
    data = x.data.flatten() if isinstance(x, Mat) else np.array(x).flatten()
    return float(scipy_stats.iqr(data))


@register("kurtosis")
def _kurtosis(x):
    data = x.data.flatten() if isinstance(x, Mat) else np.array(x).flatten()
    return float(scipy_stats.kurtosis(data))


@register("skewness")
def _skewness(x):
    data = x.data.flatten() if isinstance(x, Mat) else np.array(x).flatten()
    return float(scipy_stats.skew(data))


@register("mode")
def _mode(x, dim=None):
    data = x.data if isinstance(x, Mat) else np.array(x)
    if dim is not None:
        axis = int(dim) - 1
        result = scipy_stats.mode(data, axis=axis, keepdims=True)
        return Mat(result.mode)
    result = scipy_stats.mode(data.flatten(), keepdims=True)
    return float(result.mode[0])


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


@register("normcdf")
def _normcdf(x, mu=0, sigma=1):
    data = x.data if isinstance(x, Mat) else np.array(x)
    mu = float(mu.data.flat[0]) if isinstance(mu, Mat) else float(mu)
    sigma = float(sigma.data.flat[0]) if isinstance(sigma, Mat) else float(sigma)
    return Mat(scipy_stats.norm.cdf(data, loc=mu, scale=sigma))


@register("norminv")
def _norminv(p, mu=0, sigma=1):
    data = p.data if isinstance(p, Mat) else np.array(p)
    mu = float(mu.data.flat[0]) if isinstance(mu, Mat) else float(mu)
    sigma = float(sigma.data.flat[0]) if isinstance(sigma, Mat) else float(sigma)
    return Mat(scipy_stats.norm.ppf(data, loc=mu, scale=sigma))


@register("normpdf")
def _normpdf(x, mu=0, sigma=1):
    data = x.data if isinstance(x, Mat) else np.array(x)
    mu = float(mu.data.flat[0]) if isinstance(mu, Mat) else float(mu)
    sigma = float(sigma.data.flat[0]) if isinstance(sigma, Mat) else float(sigma)
    return Mat(scipy_stats.norm.pdf(data, loc=mu, scale=sigma))


@register("chi2cdf")
def _chi2cdf(x, v):
    data = x.data if isinstance(x, Mat) else np.array(x)
    v = int(v.data.flat[0]) if isinstance(v, Mat) else int(v)
    return Mat(scipy_stats.chi2.cdf(data, df=v))


@register("chi2inv")
def _chi2inv(p, v):
    data = p.data if isinstance(p, Mat) else np.array(p)
    v = int(v.data.flat[0]) if isinstance(v, Mat) else int(v)
    return Mat(scipy_stats.chi2.ppf(data, df=v))


@register("tcdf")
def _tcdf(x, v):
    data = x.data if isinstance(x, Mat) else np.array(x)
    v = int(v.data.flat[0]) if isinstance(v, Mat) else int(v)
    return Mat(scipy_stats.t.cdf(data, df=v))


@register("tinv")
def _tinv(p, v):
    data = p.data if isinstance(p, Mat) else np.array(p)
    v = int(v.data.flat[0]) if isinstance(v, Mat) else int(v)
    return Mat(scipy_stats.t.ppf(data, df=v))


@register("fcdf")
def _fcdf(x, v1, v2):
    data = x.data if isinstance(x, Mat) else np.array(x)
    v1 = int(v1.data.flat[0]) if isinstance(v1, Mat) else int(v1)
    v2 = int(v2.data.flat[0]) if isinstance(v2, Mat) else int(v2)
    return Mat(scipy_stats.f.cdf(data, dfn=v1, dfd=v2))


@register("finv")
def _finv(p, v1, v2):
    data = p.data if isinstance(p, Mat) else np.array(p)
    v1 = int(v1.data.flat[0]) if isinstance(v1, Mat) else int(v1)
    v2 = int(v2.data.flat[0]) if isinstance(v2, Mat) else int(v2)
    return Mat(scipy_stats.f.ppf(data, dfn=v1, dfd=v2))


@register("gamcdf")
def _gamcdf(x, a, b=1):
    data = x.data if isinstance(x, Mat) else np.array(x)
    a = float(a.data.flat[0]) if isinstance(a, Mat) else float(a)
    b = float(b.data.flat[0]) if isinstance(b, Mat) else float(b)
    return Mat(scipy_stats.gamma.cdf(data, a=a, scale=b))


@register("gaminv")
def _gaminv(p, a, b=1):
    data = p.data if isinstance(p, Mat) else np.array(p)
    a = float(a.data.flat[0]) if isinstance(a, Mat) else float(a)
    b = float(b.data.flat[0]) if isinstance(b, Mat) else float(b)
    return Mat(scipy_stats.gamma.ppf(data, a=a, scale=b))


@register("betacdf")
def _betacdf(x, a, b):
    data = x.data if isinstance(x, Mat) else np.array(x)
    a = float(a.data.flat[0]) if isinstance(a, Mat) else float(a)
    b = float(b.data.flat[0]) if isinstance(b, Mat) else float(b)
    return Mat(scipy_stats.beta.cdf(data, a=a, b=b))


@register("betainv")
def _betainv(p, a, b):
    data = p.data if isinstance(p, Mat) else np.array(p)
    a = float(a.data.flat[0]) if isinstance(a, Mat) else float(a)
    b = float(b.data.flat[0]) if isinstance(b, Mat) else float(b)
    return Mat(scipy_stats.beta.ppf(data, a=a, b=b))


@register("unifcdf")
def _unifcdf(x, a, b):
    data = x.data if isinstance(x, Mat) else np.array(x)
    a = float(a.data.flat[0]) if isinstance(a, Mat) else float(a)
    b = float(b.data.flat[0]) if isinstance(b, Mat) else float(b)
    return Mat(scipy_stats.uniform.cdf(data, loc=a, scale=b - a))


@register("unifinv")
def _unifinv(p, a, b):
    data = p.data if isinstance(p, Mat) else np.array(p)
    a = float(a.data.flat[0]) if isinstance(a, Mat) else float(a)
    b = float(b.data.flat[0]) if isinstance(b, Mat) else float(b)
    return Mat(scipy_stats.uniform.ppf(data, loc=a, scale=b - a))


@register("poisspdf")
def _poisspdf(x, lam):
    data = x.data if isinstance(x, Mat) else np.array(x)
    lam = float(lam.data.flat[0]) if isinstance(lam, Mat) else float(lam)
    return Mat(scipy_stats.poisson.pmf(data.astype(int), mu=lam))


@register("poisscdf")
def _poisscdf(x, lam):
    data = x.data if isinstance(x, Mat) else np.array(x)
    lam = float(lam.data.flat[0]) if isinstance(lam, Mat) else float(lam)
    return Mat(scipy_stats.poisson.cdf(data.astype(int), mu=lam))


@register("binocdf")
def _binocdf(x, n, p):
    data = x.data if isinstance(x, Mat) else np.array(x)
    n = int(n.data.flat[0]) if isinstance(n, Mat) else int(n)
    p = float(p.data.flat[0]) if isinstance(p, Mat) else float(p)
    return Mat(scipy_stats.binom.cdf(data.astype(int), n=n, p=p))


@register("binopdf")
def _binopdf(x, n, p):
    data = x.data if isinstance(x, Mat) else np.array(x)
    n = int(n.data.flat[0]) if isinstance(n, Mat) else int(n)
    p = float(p.data.flat[0]) if isinstance(p, Mat) else float(p)
    return Mat(scipy_stats.binom.pmf(data.astype(int), n=n, p=p))


@register("expcdf")
def _expcdf(x, mu):
    data = x.data if isinstance(x, Mat) else np.array(x)
    mu = float(mu.data.flat[0]) if isinstance(mu, Mat) else float(mu)
    return Mat(scipy_stats.expon.cdf(data, scale=mu))


@register("exppdf")
def _exppdf(x, mu):
    data = x.data if isinstance(x, Mat) else np.array(x)
    mu = float(mu.data.flat[0]) if isinstance(mu, Mat) else float(mu)
    return Mat(scipy_stats.expon.pdf(data, scale=mu))


@register("logncdf")
def _logncdf(x, mu, sigma):
    data = x.data if isinstance(x, Mat) else np.array(x)
    mu = float(mu.data.flat[0]) if isinstance(mu, Mat) else float(mu)
    sigma = float(sigma.data.flat[0]) if isinstance(sigma, Mat) else float(sigma)
    return Mat(scipy_stats.lognorm.cdf(data, s=sigma, scale=np.exp(mu)))


@register("logninv")
def _logninv(p, mu, sigma):
    data = p.data if isinstance(p, Mat) else np.array(p)
    mu = float(mu.data.flat[0]) if isinstance(mu, Mat) else float(mu)
    sigma = float(sigma.data.flat[0]) if isinstance(sigma, Mat) else float(sigma)
    return Mat(scipy_stats.lognorm.ppf(data, s=sigma, scale=np.exp(mu)))


@register("wblcdf")
def _wblcdf(x, a, b):
    data = x.data if isinstance(x, Mat) else np.array(x)
    a = float(a.data.flat[0]) if isinstance(a, Mat) else float(a)
    b = float(b.data.flat[0]) if isinstance(b, Mat) else float(b)
    return Mat(scipy_stats.weibull_min.cdf(data, c=b, scale=a))


@register("wblinv")
def _wblinv(p, a, b):
    data = p.data if isinstance(p, Mat) else np.array(p)
    a = float(a.data.flat[0]) if isinstance(a, Mat) else float(a)
    b = float(b.data.flat[0]) if isinstance(b, Mat) else float(b)
    return Mat(scipy_stats.weibull_min.ppf(data, c=b, scale=a))


@register("cdf")
def _cdf(name, x, *args):
    data = x.data if isinstance(x, Mat) else np.array(x)
    name = str(name).lower()
    dist_map = {
        "norm": (scipy_stats.norm, {"loc": args[0] if len(args) > 0 else 0, "scale": args[1] if len(args) > 1 else 1}),
        "chi2": (scipy_stats.chi2, {"df": args[0]}),
        "t": (scipy_stats.t, {"df": args[0]}),
        "f": (scipy_stats.f, {"dfn": args[0], "dfd": args[1]}),
        "gamma": (scipy_stats.gamma, {"a": args[0], "scale": args[1] if len(args) > 1 else 1}),
        "beta": (scipy_stats.beta, {"a": args[0], "b": args[1]}),
        "uniform": (scipy_stats.uniform, {"loc": args[0], "scale": args[1] - args[0]}),
        "expon": (scipy_stats.expon, {"scale": args[0]}),
    }
    if name in dist_map:
        dist, params = dist_map[name]
        return Mat(dist.cdf(data, **params))
    return Mat(np.zeros_like(data))


@register("icdf")
def _icdf(name, p, *args):
    data = p.data if isinstance(p, Mat) else np.array(p)
    name = str(name).lower()
    dist_map = {
        "norm": (scipy_stats.norm, {"loc": args[0] if len(args) > 0 else 0, "scale": args[1] if len(args) > 1 else 1}),
        "chi2": (scipy_stats.chi2, {"df": args[0]}),
        "t": (scipy_stats.t, {"df": args[0]}),
        "f": (scipy_stats.f, {"dfn": args[0], "dfd": args[1]}),
        "gamma": (scipy_stats.gamma, {"a": args[0], "scale": args[1] if len(args) > 1 else 1}),
        "beta": (scipy_stats.beta, {"a": args[0], "b": args[1]}),
        "uniform": (scipy_stats.uniform, {"loc": args[0], "scale": args[1] - args[0]}),
        "expon": (scipy_stats.expon, {"scale": args[0]}),
    }
    if name in dist_map:
        dist, params = dist_map[name]
        return Mat(dist.ppf(data, **params))
    return Mat(np.zeros_like(data))


@register("pdf")
def _pdf(name, x, *args):
    data = x.data if isinstance(x, Mat) else np.array(x)
    name = str(name).lower()
    dist_map = {
        "norm": (scipy_stats.norm, {"loc": args[0] if len(args) > 0 else 0, "scale": args[1] if len(args) > 1 else 1}),
        "chi2": (scipy_stats.chi2, {"df": args[0]}),
        "t": (scipy_stats.t, {"df": args[0]}),
        "f": (scipy_stats.f, {"dfn": args[0], "dfd": args[1]}),
        "gamma": (scipy_stats.gamma, {"a": args[0], "scale": args[1] if len(args) > 1 else 1}),
        "beta": (scipy_stats.beta, {"a": args[0], "b": args[1]}),
        "uniform": (scipy_stats.uniform, {"loc": args[0], "scale": args[1] - args[0]}),
        "expon": (scipy_stats.expon, {"scale": args[0]}),
    }
    if name in dist_map:
        dist, params = dist_map[name]
        return Mat(dist.pdf(data, **params))
    return Mat(np.zeros_like(data))


@register("random")
def _random(name, *args):
    name = str(name).lower()
    if name == "norm":
        mu = args[0] if len(args) > 0 else 0
        sigma = args[1] if len(args) > 1 else 1
        shape = tuple(int(a) for a in args[2:]) if len(args) > 2 else (1,)
        return Mat(scipy_stats.norm.rvs(loc=float(mu), scale=float(sigma), size=shape))
    elif name == "unif":
        a = args[0] if len(args) > 0 else 0
        b = args[1] if len(args) > 1 else 1
        shape = tuple(int(a) for a in args[2:]) if len(args) > 2 else (1,)
        return Mat(scipy_stats.uniform.rvs(loc=float(a), scale=float(b) - float(a), size=shape))
    return Mat(np.random.rand(1))
