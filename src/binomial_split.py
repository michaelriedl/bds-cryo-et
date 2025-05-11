import numpy as np


def binomial_split(img: np.ndarray, p: float = 0.5) -> np.ndarray:
    """
    Calculate the binomial split of an input image. The input image is
    expected to be a 2D or 3D array of uint8 values. The function will
    convert the image to uint8 if it is not already in that format.

    Parameters
    ----------
    img : np.ndarray
        The input image.
    p : float
        The proportion of successes in the binomial distribution.
        Default is 0.5.

    Returns
    -------
    np.ndarray
        The binomial split of input image.
    """
    # Convert the image to uint8 if it is not already
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # Perform binomial data split
    return np.random.binomial(n=img, p=p, size=img.shape).astype(np.uint8)
