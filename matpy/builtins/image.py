"""Image Processing Toolbox for MatPy."""

import numpy as np
from matpy.builtins import register
from matpy.runtime.types import Mat


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
    plt.axis("off")
    plt.show()


@register("rgb2gray")
def _rgb2gray(img):
    data = img.data if isinstance(img, Mat) else np.array(img)
    if data.ndim == 3 and data.shape[2] >= 3:
        return Mat(
            0.2989 * data[:, :, 0] + 0.5870 * data[:, :, 1] + 0.1140 * data[:, :, 2]
        )
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
    scale = float(scale.data.flat[0]) if isinstance(scale, Mat) else float(scale)
    if data.ndim == 3:
        return Mat(zoom(data, (scale, scale, 1)))
    return Mat(zoom(data, scale))


@register("imrotate")
def _imrotate(img, angle):
    from scipy.ndimage import rotate

    data = img.data if isinstance(img, Mat) else np.array(img)
    angle = float(angle.data.flat[0]) if isinstance(angle, Mat) else float(angle)
    return Mat(rotate(data, angle, reshape=False))


@register("imcrop")
def _imcrop(img, rect=None):
    data = img.data if isinstance(img, Mat) else np.array(img)
    if rect is not None:
        r = rect.data if isinstance(rect, Mat) else np.array(rect)
        y, x, h, w = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        return Mat(data[y : y + h, x : x + w])
    return img


@register("edge")
def _edge(img, method="sobel"):
    from scipy.ndimage import sobel

    data = img.data if isinstance(img, Mat) else np.array(img)
    if data.ndim == 3:
        data = np.mean(data, axis=2)
    sx = sobel(data, axis=0)
    sy = sobel(data, axis=1)
    return Mat(np.hypot(sx, sy))


@register("imhist")
def _imhist(img, bins=256):
    import matplotlib.pyplot as plt

    data = img.data if isinstance(img, Mat) else np.array(img)
    plt.hist(data.flatten(), bins=int(bins))
    plt.title("Image Histogram")
    plt.show()


@register("histeq")
def _histeq(img, nbins=256):
    data = img.data if isinstance(img, Mat) else np.array(img)
    if data.ndim == 3:
        data = np.mean(data, axis=2)
    data_norm = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
    hist, bins = np.histogram(data_norm.flatten(), bins=int(nbins), range=(0, 256))
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    result = np.interp(data_norm.flatten(), bins[:-1], cdf_normalized)
    return Mat(result.reshape(data_norm.shape))


@register("imadjust")
def _imadjust(img, low_in=0, high_in=1, low_out=0, high_out=1, gamma=1):
    data = img.data if isinstance(img, Mat) else np.array(img)
    low_in = float(low_in.data.flat[0]) if isinstance(low_in, Mat) else float(low_in)
    high_in = (
        float(high_in.data.flat[0]) if isinstance(high_in, Mat) else float(high_in)
    )
    low_out = (
        float(low_out.data.flat[0]) if isinstance(low_out, Mat) else float(low_out)
    )
    high_out = (
        float(high_out.data.flat[0]) if isinstance(high_out, Mat) else float(high_out)
    )
    gamma = float(gamma.data.flat[0]) if isinstance(gamma, Mat) else float(gamma)
    data_norm = (data - low_in) / (high_in - low_in)
    data_norm = np.clip(data_norm, 0, 1)
    data_gamma = np.power(data_norm, gamma)
    result = data_gamma * (high_out - low_out) + low_out
    return Mat(result)


@register("medfilt2")
def _medfilt2(img, kernel_size=3):
    from scipy.ndimage import median_filter

    data = img.data if isinstance(img, Mat) else np.array(img)
    kernel_size = (
        int(kernel_size.data.flat[0])
        if isinstance(kernel_size, Mat)
        else int(kernel_size)
    )
    return Mat(median_filter(data, size=kernel_size))


@register("imgaussfilt")
def _imgaussfilt(img, sigma=1):
    from scipy.ndimage import gaussian_filter

    data = img.data if isinstance(img, Mat) else np.array(img)
    sigma = float(sigma.data.flat[0]) if isinstance(sigma, Mat) else float(sigma)
    return Mat(gaussian_filter(data, sigma=sigma))


@register("imfilter")
def _imfilter(img, h):
    from scipy.ndimage import convolve

    data = img.data if isinstance(img, Mat) else np.array(img)
    h_data = h.data if isinstance(h, Mat) else np.array(h)
    return Mat(convolve(data, h_data))


@register("imerode")
def _imerode(img, se=None):
    from scipy.ndimage import binary_erosion, grey_erosion

    data = img.data if isinstance(img, Mat) else np.array(img)
    if data.dtype == bool:
        return Mat(binary_erosion(data))
    return Mat(grey_erosion(data))


@register("imdilate")
def _imdilate(img, se=None):
    from scipy.ndimage import binary_dilation, grey_dilation

    data = img.data if isinstance(img, Mat) else np.array(img)
    if data.dtype == bool:
        return Mat(binary_dilation(data))
    return Mat(grey_dilation(data))


@register("imopen")
def _imopen(img, se=None):
    from scipy.ndimage import binary_opening, grey_opening

    data = img.data if isinstance(img, Mat) else np.array(img)
    if data.dtype == bool:
        return Mat(binary_opening(data))
    return Mat(grey_opening(data))


@register("imclose")
def _imclose(img, se=None):
    from scipy.ndimage import binary_closing, grey_closing

    data = img.data if isinstance(img, Mat) else np.array(img)
    if data.dtype == bool:
        return Mat(binary_closing(data))
    return Mat(grey_closing(data))


@register("strel")
def _strel(shape, size=3):
    shape = str(shape).lower()
    size = int(size.data.flat[0]) if isinstance(size, Mat) else int(size)
    if shape == "disk" or shape == "circle":
        y, x = np.ogrid[-size : size + 1, -size : size + 1]
        return Mat((x**2 + y**2) <= size**2)
    elif shape == "square":
        return Mat(np.ones((size, size), dtype=bool))
    elif shape == "line":
        return Mat(np.ones((1, size), dtype=bool))
    return Mat(np.ones((size, size), dtype=bool))


@register("imbinarize")
def _imbinarize(img, threshold=None):
    data = img.data if isinstance(img, Mat) else np.array(img)
    if threshold is not None:
        t = (
            float(threshold.data.flat[0])
            if isinstance(threshold, Mat)
            else float(threshold)
        )
    else:
        t = np.mean(data)
    return Mat((data > t).astype(np.uint8))


@register("imcomplement")
def _imcomplement(img):
    data = img.data if isinstance(img, Mat) else np.array(img)
    if data.dtype == np.uint8:
        return Mat(255 - data)
    return Mat(1.0 - data)


@register("imtranslate")
def _imtranslate(img, tx, ty):
    from scipy.ndimage import shift

    data = img.data if isinstance(img, Mat) else np.array(img)
    tx = float(tx.data.flat[0]) if isinstance(tx, Mat) else float(tx)
    ty = float(ty.data.flat[0]) if isinstance(ty, Mat) else float(ty)
    return Mat(shift(data, [ty, tx]))


@register("imsharpen")
def _imsharpen(img, amount=1, radius=1):
    from scipy.ndimage import gaussian_filter

    data = img.data if isinstance(img, Mat) else np.array(img)
    amount = float(amount.data.flat[0]) if isinstance(amount, Mat) else float(amount)
    radius = float(radius.data.flat[0]) if isinstance(radius, Mat) else float(radius)
    blurred = gaussian_filter(data, sigma=radius)
    sharpened = data + amount * (data - blurred)
    return Mat(np.clip(sharpened, 0, 1 if data.max() <= 1 else 255))


@register("imnoise")
def _imnoise(img, noise_type="gaussian", *args):
    data = img.data if isinstance(img, Mat) else np.array(img)
    noise_type = str(noise_type).lower()
    if noise_type == "gaussian":
        mean = float(args[0]) if args else 0
        var = float(args[1]) if len(args) > 1 else 0.01
        noise = np.random.normal(mean, np.sqrt(var), data.shape)
        return Mat(np.clip(data + noise, 0, 1 if data.max() <= 1 else 255))
    elif noise_type == "salt & pepper" or noise_type == "salt_and_pepper":
        density = float(args[0]) if args else 0.05
        result = data.copy()
        num_salt = int(density * data.size / 2)
        coords = tuple(np.random.randint(0, d, num_salt) for d in data.shape)
        result[coords] = 1 if data.max() <= 1 else 255
        num_pepper = int(density * data.size / 2)
        coords = tuple(np.random.randint(0, d, num_pepper) for d in data.shape)
        result[coords] = 0
        return Mat(result)
    elif noise_type == "speckle":
        var = float(args[0]) if args else 0.04
        noise = np.random.randn(*data.shape) * np.sqrt(var)
        return Mat(data + data * noise)
    return Mat(data)


# ── Additional Image Processing Functions ──────────────────────


def _imhist(img, nbins=256):
    """Display histogram of image data."""
    data = img.data if isinstance(img, Mat) else np.array(img)
    nbins = int(nbins)
    hist, bin_edges = np.histogram(data.flatten(), bins=nbins)
    return Mat(hist), Mat(bin_edges)


def _histeq(img, nbins=256):
    """Enhance contrast using histogram equalization."""
    data = img.data if isinstance(img, Mat) else np.array(img)
    nbins = int(nbins)
    # Normalize to 0-1
    if data.max() > 1:
        data = data / 255.0
    # Compute histogram
    hist, bin_edges = np.histogram(data.flatten(), bins=nbins, range=(0, 1))
    # Compute CDF
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    # Equalize
    equalized = np.interp(data.flatten(), bin_edges[:-1], cdf)
    return Mat(equalized.reshape(data.shape))


def _imadjust(img, low_in=0, high_in=1, low_out=0, high_out=1, gamma=1):
    """Adjust image intensity values."""
    data = img.data if isinstance(img, Mat) else np.array(img)
    low_in = float(low_in)
    high_in = float(high_in)
    low_out = float(low_out)
    high_out = float(high_out)
    gamma = float(gamma)
    # Normalize
    data = (data - low_in) / (high_in - low_in)
    data = np.clip(data, 0, 1)
    # Apply gamma
    data = np.power(data, gamma)
    # Scale to output range
    data = data * (high_out - low_out) + low_out
    return Mat(data)


def _imbinarize(img, level=None):
    """Binarize image."""
    data = img.data if isinstance(img, Mat) else np.array(img)
    if level is None:
        # Use Otsu's method
        from skimage.filters import threshold_otsu

        level = threshold_otsu(data)
    else:
        level = float(level)
    return Mat((data > level).astype(np.uint8))


def _edge(img, method="sobel"):
    """Detect edges in image."""
    from scipy.ndimage import sobel, prewitt, laplace

    data = img.data if isinstance(img, Mat) else np.array(img)
    method = str(method).lower()
    if method == "sobel":
        sx = sobel(data, axis=0)
        sy = sobel(data, axis=1)
        return Mat(np.hypot(sx, sy))
    elif method == "prewitt":
        sx = prewitt(data, axis=0)
        sy = prewitt(data, axis=1)
        return Mat(np.hypot(sx, sy))
    elif method == "laplacian":
        return Mat(np.abs(laplace(data)))
    return Mat(data)


def _imcrop(img, rect=None):
    """Crop image."""
    data = img.data if isinstance(img, Mat) else np.array(img)
    if rect is not None:
        rect_data = rect.data if isinstance(rect, Mat) else np.array(rect)
        x, y, w, h = [int(v) for v in rect_data.flat]
        return Mat(data[y : y + h, x : x + w])
    return img


def _imresize(img, scale):
    """Resize image."""
    from scipy.ndimage import zoom

    data = img.data if isinstance(img, Mat) else np.array(img)
    if isinstance(scale, Mat):
        scale = scale.data
    if isinstance(scale, np.ndarray):
        if scale.size == 1:
            scale = float(scale.flat[0])
            return Mat(zoom(data, (scale, scale)))
        else:
            return Mat(zoom(data, tuple(scale.flat)))
    scale = float(scale)
    return Mat(zoom(data, (scale, scale)))


def _imrotate(img, angle, method="bilinear"):
    """Rotate image."""
    from scipy.ndimage import rotate

    data = img.data if isinstance(img, Mat) else np.array(img)
    angle = float(angle)
    if method == "nearest":
        order = 0
    elif method == "bilinear":
        order = 1
    else:
        order = 3
    return Mat(rotate(data, angle, reshape=False, order=order))


def _imfilter(img, h):
    """Filter image with filter kernel."""
    from scipy.ndimage import convolve

    data = img.data if isinstance(img, Mat) else np.array(img)
    h_data = h.data if isinstance(h, Mat) else np.array(h)
    return Mat(convolve(data, h_data))


def _imopen(img, se):
    """Morphological opening."""
    from scipy.ndimage import binary_opening, grey_opening

    data = img.data if isinstance(img, Mat) else np.array(img)
    se_data = se.data if isinstance(se, Mat) else np.array(se)
    if data.dtype == bool:
        return Mat(binary_opening(data, structure=se_data))
    return Mat(grey_opening(data, footprint=se_data))


def _imclose(img, se):
    """Morphological closing."""
    from scipy.ndimage import binary_closing, grey_closing

    data = img.data if isinstance(img, Mat) else np.array(img)
    se_data = se.data if isinstance(se, Mat) else np.array(se)
    if data.dtype == bool:
        return Mat(binary_closing(data, structure=se_data))
    return Mat(grey_closing(data, footprint=se_data))


def _imdilate(img, se):
    """Morphological dilation."""
    from scipy.ndimage import binary_dilation, grey_dilation

    data = img.data if isinstance(img, Mat) else np.array(img)
    se_data = se.data if isinstance(se, Mat) else np.array(se)
    if data.dtype == bool:
        return Mat(binary_dilation(data, structure=se_data))
    return Mat(grey_dilation(data, footprint=se_data))


def _imerode(img, se):
    """Morphological erosion."""
    from scipy.ndimage import binary_erosion, grey_erosion

    data = img.data if isinstance(img, Mat) else np.array(img)
    se_data = se.data if isinstance(se, Mat) else np.array(se)
    if data.dtype == bool:
        return Mat(binary_erosion(data, structure=se_data))
    return Mat(grey_erosion(data, footprint=se_data))


@register("bwlabel")
def _bwlabel(img, connectivity=8):
    """Label connected components in binary image."""
    from scipy.ndimage import label

    data = img.data if isinstance(img, Mat) else np.array(img)
    if connectivity == 8:
        structure = np.ones((3, 3))
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    labeled, num_features = label(data, structure=structure)
    return Mat(labeled), num_features


@register("regionprops")
def _regionprops(img):
    """Measure properties of image regions."""
    from scipy.ndimage import center_of_mass

    data = img.data if isinstance(img, Mat) else np.array(img)
    labels = np.unique(data)
    labels = labels[labels > 0]
    props = []
    for label_val in labels:
        mask = data == label_val
        com = center_of_mass(mask)
        props.append(
            {
                "label": int(label_val),
                "centroid": com,
                "area": int(mask.sum()),
            }
        )
    return props
