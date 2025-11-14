# import the necessary packages
import numpy as np
# import glob  <- We don't need this anymore
import cv2
import matplotlib.pyplot as plt
import sys  # <- ADD THIS IMPORT

def equalize(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

# --- START OF THE FIX ---

# Get the image path from the command line
# sys.argv[0] is the script name, sys.argv[1] is the first argument
if len(sys.argv) < 2:
    print("Error: Please provide an image path.")
    sys.exit(1) # Exit the script if no image is given

imagePath = sys.argv[1]

# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread(imagePath)

# Check if the image loaded correctly
if image is None:
    print(f"Error: Could not load image from path: {imagePath}")
    sys.exit(1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("Image loaded successfully, processing...")
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

fg = cv2.addWeighted(blurred, 1.5, gray, -0.5, 0)
kernel_sharp = np.array((
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]), dtype='int')
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian = laplacian.clip(min=0)
auto = auto_canny(fg)
auto1 = auto_canny(blurred)
im = cv2.filter2D(auto, -1, kernel_sharp)

x = laplacian.astype(np.uint8)
auto2 = auto_canny(x)
im1 = cv2.filter2D(auto2, -1, kernel=kernel_sharp)

plt.figure()
plt.title("fg")
plt.imshow(auto, cmap='gray')
plt.figure()
plt.title("Blur")
plt.imshow(auto1, cmap='gray')
plt.figure()
plt.title("laplace")
plt.imshow(laplacian, cmap='gray')
plt.figure()
plt.title("lapalace1")
plt.imshow(x, cmap='gray')
plt.figure()
plt.title("edge laplace")
plt.imshow(im1, cmap='gray')
plt.figure()
plt.imshow(im, cmap='gray')

print("Showing results... (Check for a pop-up window)")
plt.show()

# --- END OF THE FIX ---

# The old code below is not needed for this
# plt.figure("Original")
# plt.close()
# plt.figure("Nothing")
# plt.close()
# plt.figure("Blur/Smooth")
# plt.close()