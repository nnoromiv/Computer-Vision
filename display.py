#!/usr/bin/env python 

# The very first line is used to make the program able to run on Unix (Linux, macOS) systems

"""review.py -- a "hello world" program for sxcv and cv2"""
import sys, sxcv, cv2

# If a filename was given on the command line, read it in.  Otherwise, use
# the test image built into the sxcv module.
if len (sys.argv) > 1:
    # sys.argv is the command line, split up into space-separated words.
    im = cv2.imread (sys.argv[1], cv2.IMREAD_ANYDEPTH)
    # You read an image in using cv2.imread, which returns None if the read fails.
    if im is None:
        print ("Couldn't read the image file '%s'!" % sys.argv[1],
               file=sys.stderr)
        exit (1)
        
    name = sys.argv[1] 
    cv2.imshow (name, im)
    cv2.waitKey (0)

else:
    im = sxcv.testimage1 ()
    name = "testimage"

# Output a one-line summary of the image.
print (sxcv.describe (im, name))
# sxcv.describe is passed the image and a title string, and it generates the output that appears in the terminal window.

# python review.py
# testimage is monochrome of size 13 rows x 10 columns with uint8 pixels.

# python review.py sx.jpg
# sx.jpg has 3 channels of size 512 rows x 512 columns with uint8 pixels.
