"""Plotting built-in functions for MatPy (matplotlib backend)."""

import numpy as np

_plt = None
_fig_counter = 0


def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _plt = plt
    return _plt


from matpy.builtins import register  # noqa: E402
from matpy.runtime.types import Mat  # noqa: E402


@register("figure")
def _figure(n=None):
    global _fig_counter
    plt = _get_plt()
    _fig_counter += 1
    if n is not None:
        fig = plt.figure(int(n))  # noqa: F841
    else:
        fig = plt.figure(_fig_counter)  # noqa: F841
    return _fig_counter


@register("clf")
def _clf():
    plt = _get_plt()
    plt.clf()


@register("close")
def _close(fig=None):
    plt = _get_plt()
    if fig is not None:
        plt.close(int(fig))
    else:
        plt.close("all")


@register("plot")
def _plot(*args):
    plt = _get_plt()
    if len(args) == 0:
        return

    x_data = None
    y_data = None
    fmt = ""

    i = 0
    a0 = args[0]
    if isinstance(a0, Mat):
        a0 = a0.data

    if isinstance(a0, str):
        fmt = a0
        i = 1
    elif len(args) >= 2:
        a1 = args[1]
        if isinstance(a1, Mat):
            a1 = a1.data
        if isinstance(a1, str):
            y_data = a0
            fmt = a1
            i = 2
        else:
            x_data = a0
            y_data = a1
            i = 2
            if i < len(args) and isinstance(args[i], str):
                fmt = args[i]
                i += 1
    else:
        y_data = a0
        i = 1

    if x_data is not None:
        if isinstance(x_data, Mat):
            x_data = x_data.data
        x_data = np.asarray(x_data).flatten()
    if y_data is not None:
        if isinstance(y_data, Mat):
            y_data = y_data.data
        y_data = np.asarray(y_data).flatten()

    if x_data is not None and y_data is not None:
        plt.plot(x_data, y_data, fmt)
    elif y_data is not None:
        plt.plot(y_data, fmt)


@register("scatter")
def _scatter(x, y, size=None, color=None):
    plt = _get_plt()
    xd = x.data if isinstance(x, Mat) else np.asarray(x)
    yd = y.data if isinstance(y, Mat) else np.asarray(y)
    plt.scatter(xd.flatten(), yd.flatten())


@register("bar")
def _bar(x):
    plt = _get_plt()
    xd = x.data if isinstance(x, Mat) else np.asarray(x)
    plt.bar(range(len(xd.flatten())), xd.flatten())


@register("histogram")
def _histogram(x, bins=10):
    plt = _get_plt()
    xd = x.data if isinstance(x, Mat) else np.asarray(x)
    plt.hist(xd.flatten(), bins=int(bins))


@register("xlabel")
def _xlabel(text):
    plt = _get_plt()
    plt.xlabel(str(text))


@register("ylabel")
def _ylabel(text):
    plt = _get_plt()
    plt.ylabel(str(text))


@register("title")
def _title(text):
    plt = _get_plt()
    plt.title(str(text))


@register("legend")
def _legend(*args):
    plt = _get_plt()
    if args:
        plt.legend([str(a) for a in args])
    else:
        plt.legend()


@register("grid")
def _grid(state="on"):
    plt = _get_plt()
    plt.grid(state == "on")


@register("xlim")
def _xlim(*args):
    plt = _get_plt()
    if len(args) == 2:
        plt.xlim(float(args[0]), float(args[1]))


@register("ylim")
def _ylim(*args):
    plt = _get_plt()
    if len(args) == 2:
        plt.ylim(float(args[0]), float(args[1]))


@register("axis")
def _axis(setting=None):
    plt = _get_plt()
    if setting == "equal":
        plt.axis("equal")
    elif setting == "tight":
        plt.axis("tight")
    elif setting == "off":
        plt.axis("off")


@register("subplot")
def _subplot(m, n, p):
    plt = _get_plt()
    plt.subplot(int(m), int(n), int(p))


@register("saveas")
def _saveas(fig, filename, fmt=None):
    plt = _get_plt()
    fname = str(filename)
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {fname}")


@register("gcf")
def _gcf():
    plt = _get_plt()
    return plt.gcf()


@register("colorbar")
def _colorbar():
    plt = _get_plt()
    plt.colorbar()


@register("surf")
def _surf(x, y, z):
    plt = _get_plt()
    xd = x.data if isinstance(x, Mat) else np.asarray(x)
    yd = y.data if isinstance(y, Mat) else np.asarray(y)
    zd = z.data if isinstance(z, Mat) else np.asarray(z)
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xd, yd, zd, cmap="viridis")


@register("mesh")
def _mesh(x, y, z):
    plt = _get_plt()
    xd = x.data if isinstance(x, Mat) else np.asarray(x)
    yd = y.data if isinstance(y, Mat) else np.asarray(y)
    zd = z.data if isinstance(z, Mat) else np.asarray(z)
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(xd, yd, zd)


@register("contour")
def _contour(*args):
    plt = _get_plt()
    if len(args) == 1:
        z = args[0]
        zd = z.data if isinstance(z, Mat) else np.asarray(z)
        plt.contour(zd)
    elif len(args) == 3:
        x, y, z = args
        xd = x.data if isinstance(x, Mat) else np.asarray(x)
        yd = y.data if isinstance(y, Mat) else np.asarray(y)
        zd = z.data if isinstance(z, Mat) else np.asarray(z)
        plt.contour(xd, yd, zd)


@register("contourf")
def _contourf(*args):
    plt = _get_plt()
    if len(args) == 1:
        z = args[0]
        zd = z.data if isinstance(z, Mat) else np.asarray(z)
        plt.contourf(zd)
    elif len(args) == 3:
        x, y, z = args
        xd = x.data if isinstance(x, Mat) else np.asarray(x)
        yd = y.data if isinstance(y, Mat) else np.asarray(y)
        zd = z.data if isinstance(z, Mat) else np.asarray(z)
        plt.contourf(xd, yd, zd)


@register("pie")
def _pie(x, labels=None):
    plt = _get_plt()
    xd = x.data if isinstance(x, Mat) else np.asarray(x)
    xd = xd.flatten()
    if labels is not None:
        plt.pie(xd, labels=labels)
    else:
        plt.pie(xd)


@register("plot3")
def _plot3(x, y, z, *args):
    plt = _get_plt()
    xd = x.data if isinstance(x, Mat) else np.asarray(x)
    yd = y.data if isinstance(y, Mat) else np.asarray(y)
    zd = z.data if isinstance(z, Mat) else np.asarray(z)
    fig = plt.gcf()
    ax = fig.add_subplot(111, projection="3d")
    fmt = ""
    if args and isinstance(args[0], str):
        fmt = args[0]
    ax.plot(xd.flatten(), yd.flatten(), zd.flatten(), fmt)


@register("stem")
def _stem(x, y=None):
    plt = _get_plt()
    if y is None:
        xd = x.data if isinstance(x, Mat) else np.asarray(x)
        plt.stem(xd.flatten())
    else:
        xd = x.data if isinstance(x, Mat) else np.asarray(x)
        yd = y.data if isinstance(y, Mat) else np.asarray(y)
        plt.stem(xd.flatten(), yd.flatten())


@register("fill")
def _fill(x, y, color="b"):
    plt = _get_plt()
    xd = x.data if isinstance(x, Mat) else np.asarray(x)
    yd = y.data if isinstance(y, Mat) else np.asarray(y)
    plt.fill(xd.flatten(), yd.flatten(), color)


@register("area")
def _area(x, y=None):
    plt = _get_plt()
    if y is None:
        xd = x.data if isinstance(x, Mat) else np.asarray(x)
        plt.fill_between(range(len(xd.flatten())), xd.flatten(), alpha=0.3)
        plt.plot(xd.flatten())
    else:
        xd = x.data if isinstance(x, Mat) else np.asarray(x)
        yd = y.data if isinstance(y, Mat) else np.asarray(y)
        plt.fill_between(xd.flatten(), yd.flatten(), alpha=0.3)
        plt.plot(xd.flatten(), yd.flatten())


@register("errorbar")
def _errorbar(x, y, err):
    plt = _get_plt()
    xd = x.data if isinstance(x, Mat) else np.asarray(x)
    yd = y.data if isinstance(y, Mat) else np.asarray(y)
    ed = err.data if isinstance(err, Mat) else np.asarray(err)
    plt.errorbar(xd.flatten(), yd.flatten(), yerr=ed.flatten())


@register("hold")
def _hold(state="on"):
    """Set hold state for overlaying plots."""
    plt = _get_plt()
    if state == "on" or state is True:
        plt.gca()
    elif state == "off" or state is False:
        plt.clf()


@register("box")
def _box(state="on"):
    plt = _get_plt()
    ax = plt.gca()
    if state == "on":
        ax.set_frame_on(True)
    elif state == "off":
        ax.set_frame_on(False)


@register("zlabel")
def _zlabel(text):
    plt = _get_plt()
    ax = plt.gca()
    ax.set_zlabel(str(text))


@register("xticks")
def _xticks(ticks, labels=None):
    plt = _get_plt()
    if isinstance(ticks, Mat):
        ticks = ticks.data.flatten()
    if labels is not None:
        plt.xticks(ticks, labels)
    else:
        plt.xticks(ticks)


@register("yticks")
def _yticks(ticks, labels=None):
    plt = _get_plt()
    if isinstance(ticks, Mat):
        ticks = ticks.data.flatten()
    if labels is not None:
        plt.yticks(ticks, labels)
    else:
        plt.yticks(ticks)


@register("text")
def _text(x, y, txt):
    plt = _get_plt()
    plt.text(float(x), float(y), str(txt))


@register("annotation")
def _annotation(text, x, y):
    plt = _get_plt()
    plt.annotate(str(text), xy=(float(x), float(y)))


@register("drawnow")
def _drawnow():
    plt = _get_plt()
    plt.draw()


# ── Additional Plotting Functions ──────────────────────────────


def _plot3(x, y, z, *args):
    """3-D line plot."""
    plt = _get_plt()

    fig = plt.gcf()
    ax = fig.add_subplot(111, projection="3d")
    xd = x.data if isinstance(x, Mat) else np.array(x)
    yd = y.data if isinstance(y, Mat) else np.array(y)
    zd = z.data if isinstance(z, Mat) else np.array(z)
    fmt = str(args[0]) if args else "-"
    ax.plot(xd.flatten(), yd.flatten(), zd.flatten(), fmt)


def _surf(x, y, z):
    """3-D surface plot."""
    plt = _get_plt()

    fig = plt.gcf()
    ax = fig.add_subplot(111, projection="3d")
    xd = x.data if isinstance(x, Mat) else np.array(x)
    yd = y.data if isinstance(y, Mat) else np.array(y)
    zd = z.data if isinstance(z, Mat) else np.array(z)
    ax.plot_surface(xd, yd, zd)


def _mesh(x, y, z):
    """3-D mesh plot."""
    plt = _get_plt()

    fig = plt.gcf()
    ax = fig.add_subplot(111, projection="3d")
    xd = x.data if isinstance(x, Mat) else np.array(x)
    yd = y.data if isinstance(y, Mat) else np.array(y)
    zd = z.data if isinstance(z, Mat) else np.array(z)
    ax.plot_wireframe(xd, yd, zd)


def _contour(x, y, z, levels=None):
    """Contour plot."""
    plt = _get_plt()
    xd = x.data if isinstance(x, Mat) else np.array(x)
    yd = y.data if isinstance(y, Mat) else np.array(y)
    zd = z.data if isinstance(z, Mat) else np.array(z)
    if levels is not None:
        plt.contour(xd, yd, zd, levels=int(levels))
    else:
        plt.contour(xd, yd, zd)


def _contourf(x, y, z, levels=None):
    """Filled contour plot."""
    plt = _get_plt()
    xd = x.data if isinstance(x, Mat) else np.array(x)
    yd = y.data if isinstance(y, Mat) else np.array(y)
    zd = z.data if isinstance(z, Mat) else np.array(z)
    if levels is not None:
        plt.contourf(xd, yd, zd, levels=int(levels))
    else:
        plt.contourf(xd, yd, zd)


@register("quiver")
def _quiver(x, y, u, v):
    """Quiver plot."""
    plt = _get_plt()
    xd = x.data if isinstance(x, Mat) else np.array(x)
    yd = y.data if isinstance(y, Mat) else np.array(y)
    ud = u.data if isinstance(u, Mat) else np.array(u)
    vd = v.data if isinstance(v, Mat) else np.array(v)
    plt.quiver(xd, yd, ud, vd)


@register("scatter3")
def _scatter3(x, y, z, s=None, c=None):
    """3-D scatter plot."""
    plt = _get_plt()

    fig = plt.gcf()
    ax = fig.add_subplot(111, projection="3d")
    xd = x.data if isinstance(x, Mat) else np.array(x)
    yd = y.data if isinstance(y, Mat) else np.array(y)
    zd = z.data if isinstance(z, Mat) else np.array(z)
    ax.scatter(xd.flatten(), yd.flatten(), zd.flatten())


@register("bar3")
def _bar3(x):
    """3-D bar plot."""
    plt = _get_plt()

    fig = plt.gcf()
    ax = fig.add_subplot(111, projection="3d")
    data = x.data if isinstance(x, Mat) else np.array(x)
    ax.bar(range(len(data.flatten())), data.flatten())


@register("stem3")
def _stem3(x, y, z):
    """3-D stem plot."""
    plt = _get_plt()

    fig = plt.gcf()
    ax = fig.add_subplot(111, projection="3d")
    xd = x.data if isinstance(x, Mat) else np.array(x)
    yd = y.data if isinstance(y, Mat) else np.array(y)
    zd = z.data if isinstance(z, Mat) else np.array(z)
    ax.stem(xd.flatten(), yd.flatten(), zd.flatten())


@register("waterfall")
def _waterfall(x, y, z):
    """Waterfall plot."""
    plt = _get_plt()

    fig = plt.gcf()
    ax = fig.add_subplot(111, projection="3d")
    xd = x.data if isinstance(x, Mat) else np.array(x)
    yd = y.data if isinstance(y, Mat) else np.array(y)
    zd = z.data if isinstance(z, Mat) else np.array(z)
    ax.plot_wireframe(xd, yd, zd, rstride=1, cstride=0)


def _saveas(filename, fmt=None):
    """Save figure to file."""
    plt = _get_plt()
    filename = str(filename)
    if fmt is not None:
        plt.savefig(filename, format=str(fmt))
    else:
        plt.savefig(filename)


@register("print")
def _print(filename, fmt=None):
    """Print figure to file."""
    _saveas(filename, fmt)


@register("gca")
def _gca():
    """Get current axes."""
    plt = _get_plt()
    return plt.gca()


def _gcf():
    """Get current figure."""
    plt = _get_plt()
    return plt.gcf()


@register("axes")
def _axes(*args):
    """Create axes."""
    plt = _get_plt()
    if args:
        plt.axes(args[0])
    else:
        plt.axes()


def _subplot(m, n, p):
    """Create subplot."""
    plt = _get_plt()
    plt.subplot(int(m), int(n), int(p))
