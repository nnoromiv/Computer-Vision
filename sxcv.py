#-------------------------------------------------------------------------------
# s x c v . p y   --   programmer-friendly OpenCV wrappers for CE316/866
#-------------------------------------------------------------------------------

"""OpenCV is written in C++ and is really intended for speed of
execution.  Its Python wrappers are provided by the cv2 module and are
quite "thin", meaning that they mimic closely the C++ calls.  That means
they are less elegant than they could be for a Python programmer.
Fortunately, the image representation used by OpenCV in Python is that
of a numpy array, so we are able to build on both its and OpenV's
functionality.

This module wraps some OpenCV and numpy functionality to make it more
convenient for a programmer -- or Computer Vision student -- to use.
The functionality it provides as supplied is actually quite limited
because the intention is that YOU add further routines to it to provide
specific new functionality.  For each new piece of functionality to be
added, you will be given the specification of the routine and some
tests, supplied as text in its introductory comment, and you add code to
do the processing -- the first laboratory script gives an example.  You
test that you have the functionality right by typing the shell command:

   python -m doctest sxcv.py

This pulls out all of the tests from the comments in the file and
determines whether the routines produce the expected outputs or not.
Successes, cases where the output is as expected, are normally not
reported but failures are reported.  Adding the "-v" qualifier to the
above command makes the doctest module output information about each
test.

With the necessary functions written, you will be able to integrate them
into complete programs that do useful computer vision tasks in the
second half of the laboratory programme.

The easiest way to read through all the documentation in this file is by
typing the command:

   pydoc sxcv

For the interested reader, the documentation at the top of each routine
uses Google's style of Python docstrings because (to the author's eye)
it is the easiest to type and the most elegant-looking when printed out
unprocessed.  I wish those in charge of the Python language would
specify what we should all use!

"""

#-------------------------------------------------------------------------------
# Boilerplate.
#-------------------------------------------------------------------------------

import sys, os, platform, tempfile
import cv2, numpy
import matplotlib.pylab as plt

#-------------------------------------------------------------------------------
# MODULE INITIALIZATION.
#-------------------------------------------------------------------------------

# Set the default values of global variables.
DEBUG = False

# We occasionally have to do things differently on different operating systems,
# so figure out what we're running on.
systype = platform.system ()

# Extract any settings from the environment variable "SXCV" and store them in
# the global list ENVIRONMENT.
key =  "SXCV"
if key in os.environ:
    val = os.environ[key].lower ()
    ENVIRONMENT = val.split ()
    # Set our globals according to keywords in the environment variable.
    if "debug" in ENVIRONMENT: DEBUG = True
else:
    ENVIRONMENT = []

#-------------------------------------------------------------------------------
# DEBUGGING SUPPORT.
#-------------------------------------------------------------------------------
# The library is able to provide some limited debugging information for users.
# Our "debug mode" can be turned on or off explicitly by the program, and the
# neatest way of doing that is to support a "-debug" command-line qualifier in
# programs, invoking sx.debug_on() if it was provided.  However, even for
# programs that do not do this, we can enable debugging mode by setting the
# environment variable "SXCV" to include the space-separated word "debug".
# To set the environment variable on Linux or a Mac, use something like
#    export SXCV="debug sixel"
# in your "~/.bashrc" file (for Bash) or your "~/.zshrc" file (for Z-shell).
# On Windows, you would type
#    set SXCV="debug"
# to the command prompt.

def debug_set (value):
    """
    Set the value of our debugging state.

    Args:
        value (bool): value to which the state should be set
    """
    global DEBUG

    DEBUG = value

#-------------------------------------------------------------------------------
def debugging ():
    """
    Return the debugging state.

    Args:
        none

    Returns:
        bool: whether debugging is enabled
    """
    global DEBUG

    return DEBUG

#-------------------------------------------------------------------------------
def debug_off ():
    """
    Turn off debugging.

    Args:
        none
    """
    debug_set (False)

#-------------------------------------------------------------------------------
def debug_on ():
    """
    Turn on debugging.

    Args:
        none
    """
    debug_set (True)

#-------------------------------------------------------------------------------
def ddisplay (im, title, delay=0, destroy=True):
    """Display an image when debugging.

    Args:
        im (image): image to be displayed
        title (str): information about what is being displayed
        delay (int): number of ms to display it for, or zero to wait for
                     a keypress (default: 0)
        destroy (bool): whether or not the window should be destroyed
                        after displaying (default: True)
    """
    global DEBUG, ENVIRONMENT

    if debugging ():
        display (im, title, delay, destroy)

#-------------------------------------------------------------------------------
# SUPPORT ROUTINES.
#-------------------------------------------------------------------------------

def arrowhead ():
    """
    Return the arrowhead image in the software chapter of the lecture notes.

    Args:
        none

    Returns:
        im (image): numpy structure containing the image

    Tests:
        >>> im = arrowhead ()
        >>> im.shape
        (10, 9)
        >>> print (im[0,2])
        0
        >>> print (im[6,3])
        0
        >>> print (im[3,6])
        255
    """
    im = numpy.array ([
        [0,   0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0, 255,   0,   0,   0,   0],
        [0,   0,   0, 255, 255, 255,   0,   0,   0],
        [0,   0, 255, 255, 255, 255, 255,   0,   0],
        [0,   0,   0,   0, 255,   0,   0,   0,   0],
        [0,   0,   0,   0, 255,   0,   0,   0,   0],
        [0,   0,   0,   0, 255,   0,   0,   0,   0],
        [0,   0,   0,   0, 255,   0,   0,   0,   0],
        [0,   0,   0,   0, 255,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,   0,   0,   0]
    ], dtype="uint8")
    return im

#-------------------------------------------------------------------------------
def create_mask (name):
    """
    Return one of the commonly-used convolution masks."

    Args:
        name (str): name of the mask to be generated, one of:
                    blur3, blur5, laplacian

    Returns:
        im (image): numpy array containing the mask values

    Raises:
         ValueError: when invoked with an unsupported name

    Tests:
        >>> mask = create_mask ("blur3")
        >>> print (mask)
        [[1 1 1]
         [1 1 1]
         [1 1 1]]

        >>> mask = create_mask ("blur5")
        >>> print (mask)
        [[1 1 1 1 1]
         [1 1 1 1 1]
         [1 1 1 1 1]
         [1 1 1 1 1]
         [1 1 1 1 1]]

        >>> mask = create_mask ("laplacian")
        >>> print (mask)
        [[ 1  1  1]
         [ 1 -8  1]
         [ 1  1  1]]

        >>> mask = create_mask ("whatsit")
        Traceback (most recent call last):
         ...
        ValueError: I don't know how to generate a 'whatsit' mask!
    """
    # ASIDE: One of the reasons for having this routine is to show how an
    # exception in a test is handled -- the last case above does it and the
    # exception is triggered in the trailing else case below.

    if name == "blur3":
        im = numpy.array ([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ], dtype="int")

    elif name == "blur5":
        im = numpy.array ([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ], dtype="int")

    elif name == "laplacian":
        im = numpy.array ([
            [1,  1, 1],
            [1, -8, 1],
            [1,  1, 1]
        ], dtype="int")

    else:
        # We have a problem.
        raise ValueError ("I don't know how to generate a '%s' mask!" % name)

    # Return the mask we have created.
    return im

#-------------------------------------------------------------------------------
def describe (im, title="Image"):
    """
    Describe the image `im`.

    Args:
        im (image): image to be described
        title (str): name associated with the image (default: "Image")

    Returns:
        str: the description to be printed

    Raises:
        ValueError: when invoked with an invalid image

    Tests:
        >>> im = arrowhead ()
        >>> print (describe (im, "This image"))
        This image is monochrome of size 10 rows x 9 columns with uint8 pixels.
    """
    text = ""
    ns = len (im.shape)
    if ns == 2:
        # A two-element shape means a monochrome image.
        ny, nx = im.shape
        channels = "is monochrome"
    elif ns == 3:
        # A three-element shape means a multi-channel image.
        ny, nx, nc = im.shape
        channels = "has %d channels" % nc
    else:
        # We have a problem.
        raise ValueError ("I have a '%d'-dimensional image!" % ns)

    # Generate the actual description.
    text += "%s %s of size %d rows x %d columns with %s pixels." % \
        (title, channels, ny, nx, im.dtype)

    # Return the text.
    return text

#-------------------------------------------------------------------------------
def display (im, title=None, delay=0, destroy=True):
    """
    Display an image via `cv2.imshow`.

    Args:
        im (image): image to be displayed
        title (str): name of the window to display it in
                     (default: program name)
        delay (int): number of ms to display it for or zero to wait for
                     a keypress (default: 0)
        destroy (bool): whether or not the window should be destroyed
                        after displaying (default: True)
    """
    global ENVIRONMENT

    if title is None:
        title = sys.argv[0]

    # The following code allows Adrian to display images in his terminal
    # window when producing screencasts; otherwise, it pops up a OpenCV
    # window to display the image.  If you want to display images in the
    # terminal yourself, you MUST be using iTerm2 (Mac) or mlterm
    # (Linux) AND have `img2sixel` installed.
    if "sixel" in ENVIRONMENT and systype != "Windows":
        display_sixel (im, title, 256)
    else:
        cv2.imshow (title, im)
        cv2.waitKey (delay)
        if destroy:
            cv2.destroyWindow (title)

#-------------------------------------------------------------------------------
def display_sixel (im, title, levels=256):
    """
    Display `im` as sixels via the external program `img2sixel`.

    Args:
        im (image): image to be displayed
        title (str): information about what is being displayed
        levels (int): number of output levels to be produced
                      (default: 256)
    """
    # As well as being useful in its own right, this routine serves as a
    # template for any other routines that need to run an external program on
    # an OpenCV image.  The basic strategy is to save the image out as a
    # temporary ".png" file (it needs to be an uncompressed format), then run
    # a shell command on it.  In this particular case, the command does not
    # create any output but it is easy to edit the temporary filename to have
    # a different extension if the external program needs that, and then read
    # in the result of processing via cv2.imread as usual.

    # As the sixel output goes into the terminal window, output the title
    # above it so we can find it when we scroll up the window.
    print (title + ":")

    # Save the image to a temporary file, run img2sixel on it, then delete the
    # file.
    fn = tempfile.NamedTemporaryFile (suffix=".png").name
    cv2.imwrite (fn, im)
    # The following works with at least zsh.
    cmd = "img2sixel -p %d %s 2>/dev/null" % (levels, fn)
    os.system (cmd)
    os.remove (fn)

    # Terminate the line in the output in case img2sixel didn't.
    print ()

#-------------------------------------------------------------------------------
def examine (im, aty=None, atx=None, rows=15, cols=15, title=None):
    """
    Return the pixel values of a region of an image in a form
    suitable for printing out.

    Args:
        im (image): image to be examined
        aty (int): middle row of the region to be examined
                   (default: middle of image)
        atx (int): middle column of the region to be examined
                   (default: middle of image)
        rows (int): maximum number of rows to be printed (default: 15)
        cols (int): maximum number of columns to be printed (default: 15)

    Returns:
        str: the formatted output to be printed

    Tests:
        >>> im = create_mask ("laplacian")
        >>> print (examine (im)[:-1])
        [3 x 3 region of 3 x 3-pixel monochrome image at (1,1)]:
                  0   1   2
               ------------
            0|    1   1   1
            1|    1  -8   1
            2|    1   1   1
    """
    # Work out the default values of arguments.
    ny = im.shape[0]
    nx = im.shape[1]
    nc = 0 if len (im.shape) < 3 else im.shape[2]
    if aty is None: aty = ny // 2
    if atx is None: atx = nx // 2

    # Work out the region to display.
    ylo = max (aty - rows//2, 0)
    yhi = min (ylo + rows, ny)
    rows = yhi - ylo

    xlo = max (atx - cols//2, 0)
    xhi = min (xlo + cols, nx)
    cols = xhi - xlo

    # Start the output with the title and information about the region.  All
    # our output will be appended to the variable 'text'.
    text = ""
    if not title is None: text += title + "\n"
    channels = "monochrome" if nc == 0 else "%d-channel" % nc
    text += "[%d x %d region of %d x %d-pixel %s image at (%d,%d)]:\n" % \
        (rows, cols, ny, nx, channels, aty, atx)

    # Generate the header line and add it to text.
    start = "       "
    line = ""
    for x in range (xlo, xhi):
        line += "%4d" % x
    text += start + line + "\n" + start + "-" * len (line) + "\n"

    # ASIDE: A monochrome image in OpenCV has two subscripts and a colour one
    # three.  This means one cannot write a single piece of code to iterate
    # over pixels and have it work in both cases.  One can often use numpy's
    # reshape() function to make a monochrome image have three subscripts, or
    # just use whole-array operations; but there are a few occasions where you
    # need to iterate over subscripts explicitly.  This code shows you how.

    # Generate the image output.  We iterate over the rows of the image.  For
    # a monochrome image, we simply output the pixels along each row; but for
    # a colour image, we produce a row for each channel.
    for y in range (ylo, yhi):
        text += "%5d| " % y
        if nc == 0:
            # Monochrome so use two subscripts.
            for x in range (xlo, xhi):
                text += "%4d" % im[y,x]
            text += "\n"
        else:
            # Multi-channel so use three subscripts.
            for c in range (0, nc):
                if c > 0: text += start[:-2] + "| "
                for x in range (xlo, xhi):
                    text += "%4d" % im[y,x,c]
                text += "\n"

    # Return what we have produced, ready to be printed out.
    return text

#-------------------------------------------------------------------------------
def otsu (im):
    """Determine the threshold by Otsu's method.  It is intended to be used on
    monochrome (single-channel) images but will determine the threshold for a
    colour image if called with one --- whether that makes any real sense is up
    to the caller.

    Args:
        im (image): image to be examined

    Returns:
        threshold (float): the threshold as determined by Otsu's method

    Tests:
    >>> im = testimage1 ()
    >>> print (otsu (im))
    14
    """
    # Initialization.
    if len (im.shape) == 2:
        ny, nx = im.shape
        nc = 1
        im = im.reshape (ny, nx, nc)
    else:
        ny, nx, nc = im.shape
    npixels = ny * nx * nc

    # Work out the histogram.
    ngreys = int (im.max () + 1.5)  # round the value
    hist = numpy.zeros (ngreys)
    for r in range (0, ny):
        for c in range (0, nx):
            for b in range (0, nc):
                v = int (im[r,c,b] + 0.5)  # round the value
                hist[v] += 1

    # Step over all the possible thresholds, calculating the between-class
    # variance at each step and working out its maximum as we go.
    sum = im.sum ()
    sumB = totB = threshold = max_var = 0
    for t in range (0, ngreys):
        sumB += hist[t]
        if sumB == 0: continue
        sumF = npixels - sumB
        if sumF == 0: break
        totB += t * hist[t]
        mB = totB / sumB
        mF = (sum - sumB) / sumF
        var = (sumB / sum) * (sumF / sum) * (mB - mF)**2
        if var > max_var:
            max_var = var
            threshold = t
    return threshold

#-------------------------------------------------------------------------------
def plot_histogram (x, y, title, colours=["blue", "green", "red"]):
    """
    Plot a histogram (bar-chart) of the data in `x` and `y` using
    Matplotlib.  The `y` array can be either a single-dimensional one
    (for the histogram of a monochrome image) or two-dimensional for a
    colour image, in which case the first dimension selects the colour
    band and the second the value in that colour band.  `title` is the
    title of the plot, shown along its top edge.

    Args:
        x (array): numpy array containing the values to plot along the
                   abscissa (x) axis
        y (array): numpy array of the same length as `x` containing the
                   values to plot along the ordinate (y) axis
        title (str): title to put along the top edge of the plot
        colours (list of strings): the colours to use when there is more
                                   than one plot on the axes
                                   (default: blue, green, red)
    """
    # ASIDE: This routine handles monochrome and multi-channel image histogram
    # plotting in essentially the same way as examine did for images.

    # Set up the plot.
    plt.figure ()
    plt.grid ()
    plt.xlim ([0, x[-1]])
    plt.xlabel ("grey level")
    plt.ylabel ("frequency")
    plt.title (title)

    # Plot the data.
    if len (y.shape) == 1:
        plt.bar (x, y, color="grey")
    else:
        nc, np = y.shape
        for c in range (0, nc):
            plt.bar (x, y[c], color=colours[c])

    # Show the result.
    plt.show()

#-------------------------------------------------------------------------------
def testimage1 ():
    """
    Return a test image whose pixels are all in the range 10 to 63.
    It is intended to be used for testing routines for forming histograms,
    contrast stretching, thresholding and morphological operations.

    Args:
        none

    Returns:
        im (image): numpy structure containing the image

    Tests:
        >>> im = testimage1 ()
        >>> im.shape
        (13, 10)
    """
    im = numpy.array ([
        [10,   12,   11,   11,   12,   11,   10,   12,   11,   12],
        [10,   10,   10,   10,   10,   10,   10,   10,   10,   11],
        [11,   10,   14,   15,   10,   10,   10,   10,   15,   10],
        [10,   10,   14,   15,   10,   10,   10,   10,   10,   10],
        [10,   10,   14,   14,   10,   10,   10,   10,   10,   10],
        [10,   10,   10,   10,   15,   13,   10,   10,   10,   12],
        [12,   10,   10,   10,   14,   13,   10,   15,   10,   10],
        [12,   10,   10,   10,   10,   14,   10,   14,   14,   11],
        [12,   14,   14,   10,   10,   10,   10,   14,   10,   11],
        [10,   13,   14,   10,   10,   10,   15,   15,   10,   12],
        [12,   14,   15,   10,   10,   10,   10,   10,   10,   10],
        [10,   10,   10,   10,   10,   10,   10,   10,   10,   12],
        [11,   10,   11,   10,   12,   12,   11,   11,   10,   11],
    ], dtype="uint8")
    return im

#-------------------------------------------------------------------------------
def testimage3 ():
    """
    Return a three-channel colour image whose pixels are all in the range
    10 to 65.  Each colour channel of the returned image has the same pattern
    of pixels but each channel has the channel number added to every pixel.
    The image is intended to be used for testing routines for forming
    histograms, contrast stretching, thresholding, and so on.

    Args:
        none

    Returns:
        im (image): numpy structure containing the image

    Tests:
        >>> im = testimage3 ()
        >>> im.shape
        (13, 10, 3)
    """
    ch0 = testimage1 ()
    ch1 = ch0 + 1
    ch2 = ch0 + 2
    return cv2.merge ([ch0, ch1, ch2])

#-------------------------------------------------------------------------------
def version ():
    """
    Return our version, the date on which it was last edited.

    Args:
        none

    Returns:
        str: the version information as a string
    """
    global TS

    # The content of TS is updated every time Emacs saves the file.
    return TS[13:32]

#-------------------------------------------------------------------------------
# LIBRARY ROUTINES.
#-------------------------------------------------------------------------------
def mean (im):
    """
    Return the mean of the pixel values an image.

    Args:
        im (image): image for which the mean value is to be found

    Returns:
        ave (float): the mean of the image

    Tests:
        >>> ave = mean (arrowhead ())
        >>> print ("OK") if abs (ave - 39.66666666) < 1.0e-5 else print ("bad")
        OK
    """
    
    return numpy.mean(im)

def highest (im):
    """
    Return the maximum of the pixel values of an image.

    Args:
        im (image): image for which the maximum value is to be found

    Returns:
        hi (of same type as image): highest value in the image

    Tests:
        >>> im = testimage1 ()
        >>> hi = highest (im)
        >>> print (hi)
        15
        >>> print (type (hi))
        <class 'int'>
    """
    
    return int(numpy.max(im))


def lowest (im):
    """
    Return the minimum of the pixel values of an image.

    Args:
        im (image): image for which the maximum value is to be found
    
    Returns:
        lo (of same type as image): lowest value in the image

    Tests:
        >>> im = testimage1 ()
        >>> lo = lowest (im)
        >>> print (lo)
        10
        >>> print (type (lo))
        <class 'int'>
    """
    
    return int(numpy.min(im))

def extremes (im):
    """
    Return the minimum and maximum of the pixel values of an image.

    Args:
        im (image): image for which the maximum value is to be found
    
    Returns:
        lo (of same type as image): lowest value in the image
        hi (of same type as image): highest value in the image

    Tests:
        >>> im = testimage1 ()
        >>> lo, hi = extremes (im)
        >>> print (lo)
        10
        >>> print (hi)
        15
        >>> print (type (hi))
        <class 'int'>
    """
    
    return lowest(im), highest(im)


def histogram (im):
    """
    Return a histogram (frequency for each grey level) of the image `im`.
    The maximum pixel value is determined from the image this is use to
    determine the number of grey levels in the plot.

    The first value returned is an array of the grey levels.
    For a monochrome image, the second value returned is a single-index
    array containing the counts for each grey level; but for a colour
    image, it is a two-index array, with the first index selecting the
    channel (colour band) and the second the counts for that band's
    pixel values.

    Args:
        im (image): image for which the histogram is to be produced

    Returns:
        x (int): grey-level values for the abscissa (x) axis
        y (int): frequency counts for the ordinate (y) axis

    Tests:
        >>> im = testimage1 ()
        >>> x, y = histogram (im)
        >>> print (x)
        [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
        >>> print (y)
        [ 0  0  0  0  0  0  0  0  0  0 80 13 13  3 13  8]

    """
    
    levels = highest (im) + 1  # Get the number of grey levels
    vals = numpy.arange(levels) # Create an array of grey level values
        
    if len (im.shape) == 2:
    # It's a monochrome image, so we have only one histogram.
        hist, _ = numpy.histogram(im, bins=numpy.arange(levels + 1))

    else:
    # It's a multi-channel image, so we have a separate histogram for
    # each of its channels.
        ny, nx, nc = im.shape
        hist = numpy.zeros ((nc, levels), dtype=int)
        
        for c in range (nc):
            hist[c], _ = numpy.histogram(im[:, :, c], bins=numpy.arange(levels + 1))
    
    return vals, hist

def binarize (im, threshold, below=0, above=255):
    """Threshold monochrome image `im` at value `thresh`, setting pixels
    with value below `thresh` to `below` and those with larger values to
    `above`.  The resulting image is returned.

    Args:
        im (image): image to be thresholded and binarized
        thresh (float): threshold value
        below (float): value to which pixels lower than `thresh` are set
        above (float): value to which pixels greater than `thresh` are set

    Returns:
        bim (image): binarized image

    Tests:
        >>> im = testimage1 ()
        >>> bim = binarize (im, 12, 7, 25)
        >>> print (bim.dtype)
        uint8
    >>> print (bim)
    [[ 7  7  7  7  7  7  7  7  7  7]
     [ 7  7  7  7  7  7  7  7  7  7]
     [ 7  7 25 25  7  7  7  7 25  7]
     [ 7  7 25 25  7  7  7  7  7  7]
     [ 7  7 25 25  7  7  7  7  7  7]
     [ 7  7  7  7 25 25  7  7  7  7]
     [ 7  7  7  7 25 25  7 25  7  7]
     [ 7  7  7  7  7 25  7 25 25  7]
     [ 7 25 25  7  7  7  7 25  7  7]
     [ 7 25 25  7  7  7 25 25  7  7]
     [ 7 25 25  7  7  7  7  7  7  7]
     [ 7  7  7  7  7  7  7  7  7  7]
     [ 7  7  7  7  7  7  7  7  7  7]]
     
        >>> bim = binarize (im, 12, 255, 0)
        >>> print (bim.dtype)
        uint8
    >>> print (bim)
    [[255 255 255 255 255 255 255 255 255 255]
     [255 255 255 255 255 255 255 255 255 255]
     [255 255   0   0 255 255 255 255   0 255]
     [255 255   0   0 255 255 255 255 255 255]
     [255 255   0   0 255 255 255 255 255 255]
     [255 255 255 255   0   0 255 255 255 255]
     [255 255 255 255   0   0 255   0 255 255]
     [255 255 255 255 255   0 255   0   0 255]
     [255   0   0 255 255 255 255   0 255 255]
     [255   0   0 255 255 255   0   0 255 255]
     [255   0   0 255 255 255 255 255 255 255]
     [255 255 255 255 255 255 255 255 255 255]
     [255 255 255 255 255 255 255 255 255 255]]
"""
    
    # Ensure the image is a NumPy array
    # im = numpy.asarray(im)
    
    if not isinstance(im, numpy.ndarray):
        raise ValueError("Input image must be a NumPy array.")
    
    # Check if the image has multiple channels (i.e., it's color)
    if len(im.shape) == 3 and im.shape[2] > 1:
        print("Warning: Input image is not grayscale. Converting to grayscale.")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
    _, bim = cv2.threshold(im, threshold, above, cv2.THRESH_TOZERO_INV)
        
    bim[bim == 0] = threshold + 1
    bim[bim <= threshold] = below
    bim[bim == threshold + 1] = above

    return bim

def hsv_to_cv2(h, s, v):
    """Convert HSV values to the representation used in OpenCV, returning the
    result as a tuple for use in cv2.inRange.

    Args:
        h (float): hue angle in the range zero to 360 degrees
        s (float): saturation percentage (0 to 100)
        v (float): value percentage (0 to 100)

    Returns:
        hsv (tuple): a tuple of values for use with cv2.inRange

    Tests:
        >>> hsv_to_cv2(360, 100, 100)
        (180, 255, 255)
    """
    # Convert hue from 0-360 to 0-180
    h_cv2 = round(h / 2)
    
    # Convert saturation and value from 0-100 to 0-255
    s_cv2 = round(s * 2.55)
    v_cv2 = round(v * 2.55)
    
    return (h_cv2, s_cv2, v_cv2)

def colour_binarize (im, low, high, below=0, above=255, hsv=True):
    """Segment colour image `im` into foreground (with value `fg`) and
    background (with value `bg`) by colour.  `low` and `high` are lists
    giving respectively the lower and upper bounds of a 'slice' of
    colors to use for segmentation.  These should be in the same colour
    space as `im`.  The resulting image is returned.

    If `hsv` is set, the colour limits are expected to be provided in
    HSV colour space and so are converted from sensible values into
    those expected by OpenCV by routine `hsv_to_cv2` before use.

    Moreover, if the hue value in the `low` is larger than that in
    `high`, then the hue is taken to span 360 degrees -- this is useful
    when segmenting red regions.

    Args:
        im (image): colour image to be threshold and binarized
        low (list of 3 values): the lower colour bounds
        high (list of 3 values): the upper colour bounds
        below (float): value to which pixels lower than `thresh` are set
                    (default: 0)
        above (float): value to which pixels greater than `thresh` are set
                    (default: 255)
        hsv (bool): indicates whether `low` and `high` are in HSV space
                    (default: True)

    Returns:
        bim (image): binarized image

    Tests:
        >>> import numpy
        >>> im = numpy.zeros ((10,12,3))
        >>> im[5:8,5:10] = [4,50,50]
        >>> im[5:6,8:10] = [177,50,50]
        >>> low =  [350, 10, 10]
        >>> high = [ 20, 90, 90]
        >>> mask = colour_binarize (im, low, high)
        >>> print (mask)
        [[  0   0   0   0   0   0   0   0   0   0   0   0]
         [  0   0   0   0   0   0   0   0   0   0   0   0]
         [  0   0   0   0   0   0   0   0   0   0   0   0]
         [  0   0   0   0   0   0   0   0   0   0   0   0]
         [  0   0   0   0   0   0   0   0   0   0   0   0]
         [  0   0   0   0   0 255 255 255 255 255   0   0]
         [  0   0   0   0   0 255 255 255 255 255   0   0]
         [  0   0   0   0   0 255 255 255 255 255   0   0]
         [  0   0   0   0   0   0   0   0   0   0   0   0]
         [  0   0   0   0   0   0   0   0   0   0   0   0]]
        >>> mask = colour_binarize (im, low, high, above=0, below=255)
        >>> print (mask)
        [[255 255 255 255 255 255 255 255 255 255 255 255]
         [255 255 255 255 255 255 255 255 255 255 255 255]
         [255 255 255 255 255 255 255 255 255 255 255 255]
         [255 255 255 255 255 255 255 255 255 255 255 255]
         [255 255 255 255 255 255 255 255 255 255 255 255]
         [255 255 255 255 255   0   0   0   0   0 255 255]
         [255 255 255 255 255   0   0   0   0   0 255 255]
         [255 255 255 255 255   0   0   0   0   0 255 255]
         [255 255 255 255 255 255 255 255 255 255 255 255]
         [255 255 255 255 255 255 255 255 255 255 255 255]]

    """
    
    # Convert the image to CV2 if necessary
    if hsv:
        low = hsv_to_cv2(*low)
        high = hsv_to_cv2(*high)

    if hsv and low[0] > high[0]:  # Handle hue wraparound
        mask1 = cv2.inRange(im, (0, low[1], low[2]), (high[0], high[1], high[2]))
        mask2 = cv2.inRange(im, (low[0], low[1], low[2]), (360, high[1], high[2]))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(im, low, high)    
        
    bim = numpy.where(mask > 0, above, below).astype(numpy.uint8)
    
    return bim

import cv2

def largest_contour(contours):
    """Return the largest by area of a set of external contours found by OpenCV routine cv2.findContours.

    Args:
        contours (list): list of contours returned by cv2.findContours.

    Returns:
        contour (contour): the contour with the largest area.

    Tests:
        >>> im = arrowhead ()
        >>> im[1,1] = 255
        >>> c, junk = cv2.findContours (im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        >>> lc = largest_contour (c)
        >>> print (cv2.contourArea (lc))
        5.0
    """
    if not contours:
        return None  # Return None if no contours are provided

    # Find the contour with the maximum area
    return max(contours, key=cv2.contourArea)

def circularity (c):
    """Return the circularity of the contour `c`, returned from the
    OpenCV routine cv2.findContours.

    Args:
        c (contour): contours returned by cv2.findContours

    Returns:
        circ (float): the circularity of contour c

    Tests:
        >>> im = arrowhead ()
        >>> c, junk = cv2.findContours (im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        >>> lc = largest_contour (c)
        >>> print ("%.4f" % circularity (lc))
        68.3411
    """
    
    perimeter = cv2.arcLength(c, True) # Compute the perimeter
    area = cv2.contourArea(c)  # Compute the area
    
    if area == 0:
        return 0.0 # Avoid Zero division
    
    return (perimeter ** 2) / area


def rectangularity (c):
    """Return the rectangularity of the contour `c`, returned from the
    OpenCV routine cv2.findContours.

    Args:
        c (contour): contours returned by cv2.findContours

    Returns:
        rec (float): the rectangularity of contour c

    Tests:
        >>> im = arrowhead ()
        >>> c, junk = cv2.findContours (im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        >>> lc = largest_contour (c)
        >>> print ("%.4f" % rectangularity (lc))
        4.8276
    """
    
    area = cv2.contourArea(c)
    
    if area == 0:
        return 0.0 # Avoid Zero division
    
    # Compute the rotated bounding box
    rect = cv2.minAreaRect(c)  # Returns center, size (w, h), and angle of rotation
    box_area = rect[1][0] * rect[1][1]  # Width * Height of the bounding box

    if box_area == 0:
        return 0.0  # Avoid division by zero

    return box_area / area
    


#-------------------------------------------------------------------------------
# EPILOGUE.
#-------------------------------------------------------------------------------
TS = "Time-stamp: <2025-01-07 16:08:25 Adrian F Clark (alien@essex.ac.uk)>"

# Local Variables:
# time-stamp-line-limit: -10
# End:
#-------------------------------------------------------------------------------
# "That's all, folks!"
#-------------------------------------------------------------------------------
