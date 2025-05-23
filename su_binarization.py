import cv2
import numpy as np
from skimage.filters import threshold_otsu

# compute the contrast image based on local max and min in 3x3 neighborhood
def compute_contrast_image(img, eps=1e-5):
    kernel_size = 3

    # maximum filter and minimum filter (dilation and erosion)
    f_max = cv2.dilate(img, np.ones((kernel_size, kernel_size)))
    f_min = cv2.erode(img, np.ones((kernel_size, kernel_size)))

    # contrast formula from Su et al.
    contrast = (f_max - f_min) / (f_max + f_min + eps)
    contrast = np.nan_to_num(contrast, nan=0.0)  # replace NaN with 0

    # scale contrast to 0-255 and convert to uint8
    contrast_scaled = (contrast * 255).clip(0, 255).astype(np.uint8)
    return contrast_scaled

# detect high contrast pixels using Otsu's global thresholding
def detect_high_contrast(contrast_img):
    # find Otsu's threshold
    thresh = threshold_otsu(contrast_img)

    # create binary mask for high contrast pixels
    high_contrast_mask = contrast_img > thresh

    # convert boolean mask to integer (0 or 1)
    return high_contrast_mask.astype(np.uint8)

# estimate window size and minimum number of points
def estimate_window_size(contrast_img):
    high_contrast = detect_high_contrast(contrast_img)

    # find list of pixel coordinates (y, x) where text edges were detected
    points = np.argwhere(high_contrast == 1)
    # if image is very empty, return default values
    if len(points) < 2:
        return 15, 5

    # sort points by row and measure distances between neighboring points on the same text line
    points = points[np.argsort(points[:, 0])]
    distances = []
    for i in range(1, len(points)):
        if points[i-1, 0] == points[i, 0]: # same row
            dist = abs(points[i,1] - points[i-1,1])
            if dist > 0:
                distances.append(dist)

    # if no valid horizontal distances (blank page, isolated points) return default values
    if len(distances) == 0:
        return 15, 5

    # estimate stroke width
    stroke_width = int(np.median(distances))

    # calculate window size with Su et al. formula
    window_size = max(3 * stroke_width, 15)
    if window_size % 2 == 0:
        window_size += 1  # make odd to ensure center pixel

    # estimate N_min
    if stroke_width <= 2:
        N_min = 5  # for tiny strokes we need more points to trust classification
    elif stroke_width <= 5:
        N_min = 3 # for normal strokes we use moderate number of points
    else:
        N_min = 2 # for thick strokes we can use less points

    return window_size, N_min

# perform Su et al. binarization on a grayscale document image
def su_binarization(img):
    # build the contrast image
    contrast_img = compute_contrast_image(img)

    # estimate window size and N_min dynamically
    window_size, N_min = estimate_window_size(contrast_img)
    print(f"Estimated window_size: {window_size}, N_min: {N_min}")

    # build the contrast image
    E = detect_high_contrast(contrast_img)

    # pad the images to handle edges when sliding window
    padded_img = cv2.copyMakeBorder(img, window_size//2, window_size//2,
                                    window_size//2, window_size//2, cv2.BORDER_REFLECT)
    padded_E = cv2.copyMakeBorder(E, window_size//2, window_size//2,
                                window_size//2, window_size//2, cv2.BORDER_REFLECT)

    # create an empty image to store binarized result (same size as input)
    bin_img = np.zeros_like(img, dtype=np.uint8)

    # iterate over the image with a sliding window
    height, width = img.shape
    for y in range(height):
        for x in range(width):
            x0, x1 = x, x + window_size
            y0, y1 = y, y + window_size

            # extract the patch
            patch = padded_img[y0:y1, x0:x1] # intensity values
            patch_E = padded_E[y0:y1, x0:x1] # high contrast mask values (0 or 1)

            # count number of high contrast points in the window
            Ne = np.sum(patch_E == 1)

            # if there are enough high contrast points:
            if Ne >= N_min:
                # take the pixel values at those points
                vals = patch[patch_E == 1]

                # compute the mean and std of the neighboring high contrast pixels
                mean = np.mean(vals)
                std = np.std(vals)

                # pixel classification rule (equation 2 from Su et al.)
                if img[y, x] <= mean + std / 2:
                    bin_img[y, x] = 0 # foreground (text)
                else:
                    bin_img[y, x] = 255 # background
            else:
                bin_img[y, x] = 255 # not enough high contrast neighbors so assume background

    return bin_img