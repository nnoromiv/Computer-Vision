import cv2
import sxcv

# Read the image and convert to HSV
im = cv2.imread("map.jpg")
hsv_im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# Green color range
lo_green = sxcv.hsv_to_cv2(80, 50, 50)
hi_green = sxcv.hsv_to_cv2(140, 100, 100)

# Red color range (two ranges needed to cover low and high hue values)
lo_red1 = sxcv.hsv_to_cv2(0, 50, 50)
hi_red1 = sxcv.hsv_to_cv2(10, 100, 100)
lo_red2 = sxcv.hsv_to_cv2(170, 50, 50)
hi_red2 = sxcv.hsv_to_cv2(180, 100, 100)

# Dark blue color range (100°–130° with low value for darker shades)
lo_dark_blue = sxcv.hsv_to_cv2(100, 50, 10)
hi_dark_blue = sxcv.hsv_to_cv2(130, 100, 50)

# Create masks for green, red, and dark blue
mask_green = cv2.inRange(hsv_im, lo_green, hi_green)
mask_red1 = cv2.inRange(hsv_im, lo_red1, hi_red1)
mask_red2 = cv2.inRange(hsv_im, lo_red2, hi_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)
mask_dark_blue = cv2.inRange(hsv_im, lo_dark_blue, hi_dark_blue)

# Combine all masks into a single mask
mask_combined = cv2.bitwise_or(mask_green, mask_red)
mask_combined = cv2.bitwise_or(mask_combined, mask_dark_blue)

# Optional: Invert the mask if you want to focus on everything else
inverted_mask = cv2.bitwise_not(mask_combined)

sxcv.display (mask_combined)
# sxcv.display (inverted_mask)