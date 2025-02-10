import numpy as np
import sys
import cv2

def histogram(im):
    """
    Return a histogram (frequency for each grey level) of the image `im`.

    The first value returned is an array of the grey levels.
    For a monochrome image, the second value returned is a single-index
    array containing the counts for each grey level; but for a colour
    image, it is a two-index array, with the first index selecting the
    channel (colour band) and the second the counts for that band's
    pixel values.

    Args:
        im (numpy.ndarray): Image for which the histogram is to be produced.

    Returns:
        vals (numpy.ndarray): Array of grey-level values for the x-axis.
        hist (numpy.ndarray): Frequency counts for each grey level.
    """

    if im is None:
        raise ValueError("Invalid image provided!")

    levels = int(np.max(im)) + 1  # Get the number of grey levels
    vals = np.arange(levels)  # Create an array of grey level values

    if len(im.shape) == 2:
        # Grayscale image
        hist, _ = np.histogram(im, bins=np.arange(levels + 1))
    else:
        # Color image with multiple channels
        nc = im.shape[2]
        hist = np.zeros((nc, levels), dtype=int)
        for c in range(nc):
            hist[c], _ = np.histogram(im[:, :, c], bins=np.arange(levels + 1))

    return vals, hist

if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)  # Read the image in its original format

        if im is None:
            print(f"Couldn't read the image file '{filename}'!", file=sys.stderr)
            sys.exit(1)

        vals, hist = histogram(im)
        print("Grey levels:", vals)
        print("Histogram:", hist)
    else:
        print("Usage: python script.py <image_path>")
        sys.exit(1)