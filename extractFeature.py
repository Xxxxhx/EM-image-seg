import numpy as np


def norm_and_extract_features(img, rgb_weight=0.4):

    x, y = img.shape[:-1]
    weight_arr = [rgb_weight, 1 - rgb_weight]  # [rgb_weight, x_y_coordinate weight]

    feature = np.ndarray(shape=(x * y, 5), dtype=float)  # feature: [[r,g,b,x,y]* n_pixels]

    count = 0
    scaled_color = img / img.max()
    for i in range(x):
        for j in range(y):
            position = np.array([i / x, j / y])
            a = np.concatenate((scaled_color[i][j] * weight_arr[0], position * weight_arr[1]))
            feature[count] = a
            count += 1

    # normalize to zero mean for gaussian process.
    feature = feature - np.mean(feature, axis=0)
    return feature
