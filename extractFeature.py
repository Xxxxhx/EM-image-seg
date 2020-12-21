import numpy as np
import cv2

def norm_and_extract_features(img, rgb_weight=0.3, texture_weight=0.3):

    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab = np.split(img, 3, axis=2)
    x, y = img.shape[:-1]
    lab[0] = lab[0].reshape(x, y)
    lab[1] = lab[1].reshape(x, y)
    lab[2] = lab[2].reshape(x, y)

    grad_l = np.gradient(lab[0])
    grad_l_x_max = np.max(grad_l[0])
    grad_l_y_max = np.max(grad_l[1])

    weight_arr = [rgb_weight, texture_weight, 1 - rgb_weight - texture_weight]  # [rgb_weight, x_y_coordinate weight]

    feature = np.ndarray(shape=(x * y, 7), dtype=float)  # feature: [[r,g,b,x,y]* n_pixels]

    count = 0
    scaled_color = img / img.max()


    for i in range(x):
        for j in range(y):
            position = np.array([i / x, j / y])
            grad = np.array([grad_l[0][i][j] / grad_l_x_max, grad_l[1][i][j] / grad_l_y_max])
            a = np.concatenate((scaled_color[i][j] * weight_arr[0], grad * weight_arr[1], position * weight_arr[2]))
            feature[count] = a
            count += 1

    # normalize to zero mean for gaussian process.
    feature = feature - np.mean(feature, axis=0)
    return feature
