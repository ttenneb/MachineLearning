from sklearn.ensemble import RandomForestClassifier
import cv2
import numpy as np
from matplotlib import pyplot as plt
from classification_tree import ClassificationTree
from final_data_processing import get_original_data, quantize_indexs
import pickle



# def extract_subarrays(matrix, height, width, step_size_h, step_size_w):
#     D_h, D_w, _ = matrix.shape
#     subarrays = []
#     for i in range(0, D_h - height + 1, step_size_h):
#         for j in range(0, D_w - width + 1, step_size_w):
#             subarray = matrix[i:i+height, j:j+width, :]
#             subarrays.append(subarray)
#     return subarrays

# get_original_data()
# clf = ClassificationTree()
# clf.train()
# _, acc = clf.predict()
# print(acc)

# quit()
patch_size = 10
# TODO
def sample_pixels_with_noise(image, x, y, n, colors, p):
    image = image[x-patch_size:x+patch_size, y-patch_size:y+patch_size, :]

    # Calculate the distances from the pixel at (x, y) to all other pixels
    x_coords, y_coords = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    distances = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
    # Convert the distances to probabilities
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    distances = (max_distance - distances) / (max_distance - min_distance)
    
    # Normalize the probabilities to sum to 1
    probabilities = distances / np.sum(distances)
    
    # Sample n pixels using the probabilities as weights
    indices = np.random.choice(np.arange(image.size/3), size=n, p=probabilities.flatten()).astype(np.uint8)
    pixels = image.reshape(-1, 3)[indices, :]
    
    # Replace each pixel with a randomly chosen color from the list with probability p
    new_pixels = []
    for pixel in pixels:
        if np.random.random() < p:
            idx = np.random.choice(np.arange(len(colors)))
            new_pixels.append(colors[idx])
        else:
            new_pixels.append(pixel)
    
    return new_pixels
# TODO
def sample_pixels(image, x, y, n):
    # Calculate the distances from the pixel at (x, y) to all other pixels
    image = image[x-patch_size:x+patch_size, y-patch_size:y+patch_size, :]
    x_coords, y_coords = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    distances = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
    
    # Convert the distances to probabilities
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    distances = (max_distance - distances) / (max_distance - min_distance)
    
    # Normalize the probabilities to sum to 1
    probabilities = distances / np.sum(distances)
    
    # Sample n pixels using the probabilities as weights
    indices = np.random.choice(np.arange(image.size/3), size=n, p=probabilities.flatten()).astype(np.uint8)
    pixels = image.reshape(-1, 3)[indices, :]
    
    return pixels


def generate_dataset(image, n, size, bins, p=0.1):
    # Create empty arrays for the input and output
    X = np.empty((size, n, 3))
    Y = np.empty((size, ))
    
    # change the image into a quantized image of indexs
    quantized_image = quantize_indexs(image, bins)
    
    # Generate size number of data points
    for i in range(size):
        x, y = np.random.randint(0, image.shape[0]), np.random.randint(0, image.shape[1])
        while x < patch_size  or x + patch_size > image.shape[0] or y < patch_size or y + patch_size > image.shape[1]:
            x, y = np.random.randint(0, image.shape[0]), np.random.randint(0, image.shape[1])

        # Sample n pixels from the image using the sample_pixels function
        pixels = np.array(sample_pixels_with_noise(image, x, y, n, bins, p))
        pixels = np.stack(pixels)
        # Add the sampled pixels and the pixel at (x, y) to the input and output arrays
        X[i, :, :] = np.array(pixels)
        Y[i] = quantized_image[x, y]
    
    return X, Y


im = get_original_data()

try:
    bins = pickle.load(open("bins.p", "rb"))
except:
    bins = None
    
# passing random bin to each pixel
# for i in range(301):
#     for j in range(301):
#         im[300+i, 300+j, :] = bins[np.random.randint(0, len(bins))]
print(bins.shape)
plt.imshow(bins.reshape(4, 4, 3))
plt.show()

clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0, verbose=1, n_jobs=4, bootstrap=True)
sample_size = 32
X, Y = generate_dataset(im, sample_size, 200000, bins, p=0.25)

dataset = np.concatenate((X.reshape(-1, sample_size*3), Y.reshape(-1, 1)), axis=1)

np.savetxt("dataset.csv", dataset, delimiter=",")


print(X.shape, Y.shape)
clf.fit(X.reshape(-1, sample_size*3), Y)


im = cv2.imread("Leaves_Masked.jpg")
im = np.flip(im, axis=-1)


# try and rebuild the image
for i in range(301):
    for j in range(301):
        y_hat = clf.predict(sample_pixels(im, 300+i, 300+j, sample_size).reshape(1, -1))
        im[300+i, 300+j, :] = bins[int(y_hat)]

plt.imshow(im)
plt.show()
# parts = np.array_split(im, 3)
# parts = np.stack(parts)
# parts = np.array_split(parts, 3, axis=2)
# parts = np.stack(parts)
# plt.imshow(parts[0,0, :, :, :])
# plt.show()

# down_sample = data[0:im.shape[0]:9, 0:im.shape[1]:9, :]
# print(down_sample.shape)

# plt.imshow(down_sample)
# plt.show()

# down_sample = im[0:im.shape[0]:9, 0:im.shape[1]:9, :]
# print(down_sample.shape)

# plt.imshow(down_sample)
# plt.show()


