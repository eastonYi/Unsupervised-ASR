import numpy as np
import skimage.color as color
import skimage.transform as transform
import skimage.io as iio


rgb2gray = color.rgb2gray
gray2rgb = color.gray2rgb

imresize = transform.resize
imrescale = transform.rescale

def imread(path, as_gray=False, **kwargs):
    """Return a float64 image in [-1.0, 1.0]."""
    image = iio.imread(path, as_gray, **kwargs)
    if image.dtype == np.uint8:
        image = image / 127.5 - 1
    elif image.dtype == np.uint16:
        image = image / 32767.5 - 1
    elif image.dtype in [np.float32, np.float64]:
        image = image * 2 - 1.0
    else:
        raise Exception("Inavailable image dtype: %s!" % image.dtype)
    return image


def imwrite(image, path, quality=95, **plugin_args):
    """Save a [-1.0, 1.0] image."""
    iio.imsave(path, im2uint(image), quality=quality, **plugin_args)


def imshow(image):
    """Show a [-1.0, 1.0] image."""
    iio.imshow(im2uint(image))


def immerge(images, n_rows=None, n_cols=None, padding=0, pad_value=0):
    """Merge images to an image with (n_rows * h) * (n_cols * w).

    Parameters
    ----------
    images : numpy.array or object which can be converted to numpy.array
        Images in shape of N * H * W(* C=1 or 3).

    """
    images = np.array(images)
    n = images.shape[0]
    if n_rows:
        n_rows = max(min(n_rows, n), 1)
        n_cols = int(n - 0.5) // n_rows + 1
    elif n_cols:
        n_cols = max(min(n_cols, n), 1)
        n_rows = int(n - 0.5) // n_cols + 1
    else:
        n_rows = int(n ** 0.5)
        n_cols = int(n - 0.5) // n_rows + 1

    h, w = images.shape[1], images.shape[2]
    shape = (h * n_rows + padding * (n_rows - 1),
             w * n_cols + padding * (n_cols - 1))
    if images.ndim == 4:
        shape += (images.shape[3],)
    img = np.full(shape, pad_value, dtype=images.dtype)

    for idx, image in enumerate(images):
        i = idx % n_cols
        j = idx // n_cols
        img[j * (h + padding):j * (h + padding) + h,
            i * (w + padding):i * (w + padding) + w, ...] = image

    return img


def _check(images, dtypes, min_value=-np.inf, max_value=np.inf):
    # check type
    assert isinstance(images, np.ndarray), '`images` should be np.ndarray!'

    # check dtype
    dtypes = dtypes if isinstance(dtypes, (list, tuple)) else [dtypes]
    assert images.dtype in dtypes, 'dtype of `images` shoud be one of %s!' % dtypes

    # check nan and inf
    assert np.all(np.isfinite(images)), '`images` contains NaN or Inf!'

    # check value
    if min_value not in [None, -np.inf]:
        l = '[' + str(min_value)
    else:
        l = '(-inf'
        min_value = -np.inf
    if max_value not in [None, np.inf]:
        r = str(max_value) + ']'
    else:
        r = 'inf)'
        max_value = np.inf
    assert np.min(images) >= min_value and np.max(images) <= max_value, \
        '`images` should be in the range of %s!' % (l + ',' + r)


def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    """Transform images from [-1.0, 1.0] to [min_value, max_value] of dtype."""
    _check(images, [np.float32, np.float64], -1.0, 1.0)
    dtype = dtype if dtype else images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)


def float2im(images):
    """Transform images from [0, 1.0] to [-1.0, 1.0]."""
    _check(images, [np.float32, np.float64], 0.0, 1.0)
    return images * 2 - 1.0


def float2uint(images):
    """Transform images from [0, 1.0] to uint8."""
    _check(images, [np.float32, np.float64], -0.0, 1.0)
    return (images * 255).astype(np.uint8)


def im2uint(images):
    """Transform images from [-1.0, 1.0] to uint8."""
    return to_range(images, 0, 255, np.uint8)


def im2float(images):
    """Transform images from [-1.0, 1.0] to [0.0, 1.0]."""
    return to_range(images, 0.0, 1.0)


def uint2im(images):
    """Transform images from uint8 to [-1.0, 1.0] of float64."""
    _check(images, np.uint8)
    return images / 127.5 - 1.0


def uint2float(images):
    """Transform images from uint8 to [0.0, 1.0] of float64."""
    _check(images, np.uint8)
    return images / 255.0


def cv2im(images):
    """Transform opencv images to [-1.0, 1.0]."""
    images = uint2im(images)
    return images[..., ::-1]


def im2cv(images):
    """Transform images from [-1.0, 1.0] to opencv images."""
    images = im2uint(images)
    return images[..., ::-1]
