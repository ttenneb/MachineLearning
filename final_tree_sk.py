from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree
import cv2
import numpy as np
from matplotlib import pyplot as plt
from classification_tree import ClassificationTree
from final_data_processing import get_original_data, quantize_indexs
import pickle

# get_original_data()
# clf = ClassificationTree()
# clf.train()
# _, acc = clf.predict()
# print(acc)

# quit()
patch_size = 4
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
    if p > 0:
        for pixel in pixels:
            if np.random.random() < p:
                idx = np.random.choice(np.arange(len(colors)))
                new_pixels.append(colors[idx])
            else:
                new_pixels.append(pixel)
        return np.array(new_pixels)
    else:
        return pixels
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


def alternating(matrix, func, out, bins ,sample_size, noise):
    l = 0
    r = matrix.shape[0] -1
    org = out
    while l <= r:
        for element in matrix[l, l:r+1, :]:
            x,y = element[0], element[1]
            y_hat = func(sample_pixels_with_noise(org, x, y, sample_size, bins, noise).reshape(1, -1))
            out[x, y] = bins[int(y_hat)]
        for element in matrix[r, l:r+1, :]:
           x,y = element[0], element[1]
           y_hat = func(sample_pixels_with_noise(org, x, y, sample_size, bins, noise).reshape(1, -1))
           out[x, y] = bins[int(y_hat)]
        for element in matrix[l:r+1, l, :]:
           x,y = element[0], element[1]
           y_hat = func(sample_pixels_with_noise(org, x, y, sample_size, bins, noise).reshape(1, -1))
           out[x, y] = bins[int(y_hat)]
        for element in matrix[l:r+1, r, :]:
           x,y = element[0], element[1]
           y_hat = func(sample_pixels_with_noise(org, x, y, sample_size, bins, noise).reshape(1, -1))
           out[x, y] = bins[int(y_hat)]
        l += 1
        r -= 1
    return out

# TODO
def spiral(matrix, func, output, bins, sample_size):
    print(matrix.size)
    if matrix.size == 0: 
        return output
    for element in matrix[0, :, :]:
        x,y = element[0], element[1]
        y_hat = func(sample_pixels(output, x, y, sample_size).reshape(1, -1))
        output[x, y] = bins[int(y_hat)]
    spiral(np.rot90(matrix[1:, :, :]), func, output, bins, sample_size)
    return output

def main():
    im = get_original_data()

    try:
        bins = pickle.load(open("bins.p", "rb"))
        bin_sizes = pickle.load(open("bin_sizes.p", "rb"))
        print(bin_sizes)
    except:
        bins = None
        
    # passing random bin to each pixel
    # for i in range(301):
    #     for j in range(301):
    #         im[300+i, 300+j, :] = bins[np.random.randint(0, len(bins))]
    print(bins.shape)
    plt.imshow(bins.reshape(1, -1, 3))
    plt.show()

    clf = ExtraTreesClassifier(n_estimators=10, random_state=22, n_jobs=1, criterion="entropy", class_weight=dict(bin_sizes), bootstrap=True)
    sample_size = 32
    X, Y = generate_dataset(im, sample_size, 100000, bins, p=0.3)

    dataset = np.concatenate((X.reshape(-1, sample_size*3), Y.reshape(-1, 1)), axis=1)

    np.savetxt("dataset.csv", dataset, delimiter=",")

    

    print("Training")

    clf.fit(X.reshape(-1, sample_size*3), Y)

    # test
    print("testing")
    X_test, Y_test = generate_dataset(im, sample_size, 1000, bins, p=0.0)
    print("acc p = 0", clf.score(X_test.reshape(-1, sample_size*3), Y_test))
    X_test, Y_test = generate_dataset(im, sample_size, 1000, bins, p=0.2)
    print("acc p = .2", clf.score(X_test.reshape(-1, sample_size*3), Y_test))


    im = cv2.imread("Leaves_Masked.jpg")
    im = np.flip(im, axis=-1)
    for i in range(301):
        for j in range(301):
            im[300+i, 300+j, :] = bins[np.random.randint(0, len(bins))]


    x_cords, y_cords = np.meshgrid(np.arange(start=300, stop=600), np.arange(start=300, stop=600))
    grid = np.stack((x_cords, y_cords), axis=2)

    # generate Image
    print("Generating Image")

    im_org = np.copy(im)

    im_alt = alternating(grid, clf.predict, im, bins, sample_size, 0.1)
    plt.imshow(im_alt)
    plt.show()
    im_spiral = spiral(grid, clf.predict, im_org, bins, sample_size)
    plt.imshow(im_spiral)
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
main()


