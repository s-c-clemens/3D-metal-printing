# Scott Clemens
# Created to counteract distortions of DLP and optics on print design

import numpy as np
import cv2 as cv
import math
import os

def make_background(original, height=533, width=852):
    """ This takes user design and puts on black background for resizing """
    # New, blank image before sizing:
    image = np.zeros((height, width, 3), np.uint8)

    # Starting dimensions of image must be:
    # Height: 250 px
    # Width:  250 px
    # image must be 24-bit bitmap for DLP
    (h, w, _) = original.shape
    
    # paste user design onto blank background
    image[142:392, 301:551] = original[0:h, 0:w]

    return image

def radial_distort(image, amount):
    """ This corrects for radial distortion """
    (h, w, _) = image.shape

    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)
    
    scale_x = 1
    scale_y = 1
    center_x = w/2
    center_y = h/2
    radius = w/2

    # create map with the barrel pincushion distortion formula
    for y in range(h):
        delta_y = scale_y * (y - center_y)
        for x in range(w):
            # determine if pixel is within an ellipse
            delta_x = scale_x * (x - center_x)
            distance = delta_x * delta_x + delta_y * delta_y
            if distance >= (radius * radius):
                map_x[y, x] = x
                map_y[y, x] = y
            else:
                factor = 1.0
                if distance > 0.0:
                    factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), amount)
                map_x[y, x] = factor * delta_x / scale_x + center_x
                map_y[y, x] = factor * delta_y / scale_y + center_y

    distorted_image = cv.remap(image, map_x, map_y, cv.INTER_LINEAR)
    return distorted_image

def scale_image(image, scale_x=1.07, scale_y=2.139):
    """ This scales the image to the right size """
    scaled_image = cv.resize(image, (0, 0), fx=scale_x, fy=scale_y)
    return scaled_image

def contrast_image(image, contrast=5000):
    """ This contrasts the image so there is not blurriness """
    image = np.int16(image)
    image = scaled_image *(contrast/127 + 1) - contrast
    image = np.clip(image, 0, 255)
    image = np.uint8(image)
    return image

if __name__ == "__main__":
    # prompt user for the filename (must be in the same folder as this script)
    print("This program was created to counteract distortions of DLP and optics")
    print(" on print design. The user designs a negative B&W image design that is")
    print(" 250x250 px and saves it as a 24-bit bitmap. Then, this deisng is")
    print(" selected as the file. Choose distortion and contrast values below.")
    print("***************** SAVE INFORMATION BELOW THIS LINE *****************")
    file = input("Filename:  ")
    #try:
    original_image = cv.imread(file)
    image = make_background(original_image)
    #except:
    #    print("There was an error importing the image, please try again.")
    #    exit()
    # negative values produce pincushion (inner distort)
    # positive values produce barrel (outer distort)
    amount = float(input("Distortion coefficient:  "))
    contrast = int(input("Contrast coefficient:  "))
    scale_x = 1.07
    scale_y = 2.139
    
    distorted_image = radial_distort(image, amount)
    scaled_image = scale_image(distorted_image, scale_x, scale_y)
    final_image = contrast_image(scaled_image, contrast)

    (h, w, _) = original_image.shape
    (distorted_h, distorted_w, _) = distorted_image.shape
    (scaled_h, scaled_w, _) = scaled_image.shape

    print("\n \tOriginal:\tCorrected:")
    print("Height:  ", h, "px  \t", scaled_h, "px")
    print("Width:   ", w, "px  \t", scaled_w, "px")
    print("Scale H: ", 1, "    \t", scale_y)
    print("Scale W: ", 1, "    \t", scale_x)
    # save the result
    cv.imwrite(file + '_corrected.bmp', final_image)

    # show the result
    cv.imshow('Original', original_image)
    cv.imshow('Distorted-Scaled', scaled_image)
    print("***************** SAVE INFORMATION ABOVE THIS LINE *****************")
    print("Original and corrected images displayed")
    
    cv.waitKey(0)
    cv.destroyAllWindows()
