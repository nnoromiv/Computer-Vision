import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_biscuit_image(image_path):
    # Step 1: Read the image
    im = cv2.imread(image_path)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Step 2: Binarize the image using Otsu's method
    _, bim = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 3: Find contours
    contours, _ = cv2.findContours(bim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 4: Display contour areas
    print(f"Contour areas for {image_path}:")
    for (i, c) in enumerate(contours):
        area = cv2.contourArea(c)
        print(f"  Contour {i}: Area = {area:.2f}")
    
    # Step 5: Draw all contours on the original image
    im_with_contours = im.copy()
    cv2.drawContours(im_with_contours, contours, -1, (0, 0, 255), 3)
    
    # Step 6: Display the original image, binary image, and image with contours
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    
    plt.subplot(1, 3, 2)
    plt.imshow(bim, cmap='gray')
    plt.title("Binary Image (Otsu's Threshold)")
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(im_with_contours, cv2.COLOR_BGR2RGB))
    plt.title("Contours Drawn on Image")
    
    plt.show()
    
    # Return the largest contour (the biscuit)
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

# Example usage for the three images
for image_name in ["biscuit-017.jpg", "biscuit-042.jpg", "biscuit-189.jpg"]:
    process_biscuit_image(image_name)