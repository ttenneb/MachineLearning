import cv2
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
import pickle

image_original = cv2.imread("Leaves_Masked.jpg")
# fix the colors (BGR to RGB)
image_original = np.flip(image_original, axis=-1)

# get rid of the white part so we can find the colors
image = np.array_split(image_original, 3)
image = np.stack(image)
image = np.array_split(image, 3, axis=2)
image= np.stack(image)
image = list(image.reshape(9, 300, 300, 3))
del image[4]
image = np.concatenate(np.stack(image), axis=0)

# number of bins/colors to find
bin_count = 12
min_percent = .005

show_image = False
show_palette = False
    
try:
    bins = pickle.load(open("bins.p", "rb"))
except:
    bins = None
binned_pixels = []
total_pixels = len(image.reshape(-1, 3))/9
if bins is None or len(bins) != bin_count:
    print("Defining Bins")
    # must be square
    bins = np.random.randint(high=256, low = 0, size=(bin_count, 3))
    # find nearest bin for each value

    for iteration in range(100):
        binned_pixels = [[] for i in range(bin_count)]
        print("Finding Nearest Bin for Each Value")
        # find distances for each bin for each pixel
        dist = np.sum((bins[:, None, :]-image.reshape(-1, 3))**2, axis=2)
        # get closest bin for each pixel
        idxs = np.argmin(dist, axis=0)
        # assign pixel to bin
        for i, pixel in enumerate(image.reshape(-1, 3)):
            binned_pixels[idxs[i]].append(pixel)
        
        print("Calculating Mean for Each Bin")
        # calculate mean for each bin and redefine bin value
        total_zeros = 0
        max = 0
        max_bin = 0
        
        for i, bin in enumerate(binned_pixels):
            print(len(bin)/total_pixels)
            if len(bin)/total_pixels > min_percent:
                bins[i] = np.mean(bin, axis=0)
                if len(bin) > max:
                    max = len(bin)
                    max_bin = i
            else:
                bins[i] = np.random.randint(high=256, low = 0, size=(3))
                total_zeros += 1
        if total_zeros == 0:
            break
        # stop pixels from collecting at local maximum
        bins[max_bin] = np.random.randint(high=256, low = 0, size=(3))
        print("Iteration: ", iteration, " total_zeros: ", total_zeros)        


if show_palette:
    plt.imshow(bins.reshape(1, -1, 3).astype(np.uint8))
    plt.show()

if show_image:
    # find nearest bin for each value
    dist = np.sum((bins[:, None, :]-image_original.reshape(-1, 3))**2, axis=2)
    idxs = np.argmin(dist, axis=0)

    # Map the image to the nearest colors
    quantized = bins[idxs]

    plt.imshow(quantized.reshape(image_original.shape).astype(np.uint8))
    plt.show()

pickle.dump(bins, open("bins.p", "wb"))

bin_sizes = {}


if len(binned_pixels) != 0:
    for i, bin in enumerate(binned_pixels):
        bin_sizes.update({i: len(bin)/total_pixels})

    # print(bin_sizes)
    pickle.dump(bin_sizes, open("bin_sizes.p", "wb"))

def quantize(image, bins = bins):
    # find nearest bin for each value
    dist = np.sum((bins[:, None, :]-image.reshape(-1, 3))**2, axis=2)
    idxs = np.argmin(dist, axis=0)

    # Map the image to the nearest colors
    quantized = bins[idxs]

    return quantized.reshape(image.shape).astype(np.uint8)

def quantize_indexs(image, bins = bins):
    # find nearest bin for each value
    dist = np.sum((bins[:, None, :]-image.reshape(-1, 3))**2, axis=2)
    idxs = np.argmin(dist, axis=0)

    # Map the image to the nearest colors
    quantized = idxs

    return quantized.reshape(image.shape[0], image.shape[1], 1).astype(np.uint8)

# print(bins)

def get_original_data():
    image_original = cv2.imread("Leaves_Masked.jpg")
    # fix the colors (BGR to RGB)
    image_original = np.flip(image_original, axis=-1)

    # get rid of the white part so we can find the colors
    image = np.array_split(image_original, 3)
    image = np.stack(image)
    image = np.array_split(image, 3, axis=2)
    image= np.stack(image)
    image = list(image.reshape(9, 300, 300, 3))
    del image[4]
    
    processed_data = []
    for i, image in enumerate(image):
        for r in range(4):
            processed_data.append(np.rot90(image))
           
    image = np.concatenate(np.stack(processed_data), axis=0)
    return image
