
import numpy as np
from matplotlib.image import imread
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from EM import EM


#%% md

## Read Image

#%%

file = 'images/test_image.png'
img = imread(file)
plt.imshow(img)

#%% md
## Smoothing Image

#%%

sigma = 3
img = gaussian_filter(img, sigma=(sigma, sigma, 0))
plt.imshow(img)

#%% md

## Extract Features

#%%

# Wenjuan's Part: find the optimal value of weight
weight = [1, 1]

x = img.shape[0]
y = img.shape[1]
feature = np.ndarray(shape=(x * y, 5), dtype=float)

count = 0
for i in range(x):
    for j in range(y):
        color = img[i][j] / 255
        position = np.array([i / x, j / y])
        a = np.concatenate((color * weight[0], position * weight[1]))
        feature[count] = a
        count += 1

feature = feature - np.mean(feature, axis=0)

#%% md

## Expectation-Maximization with Gaussian Mixture Model

#%%

em = EM()
res = em.fit(4, feature)

#%%

random_color = np.ndarray(shape=(4, 3))
for i in range(random_color.shape[0]):
    random_color[i] = np.array([np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)])

clustered_img = np.ndarray(shape=(x, y, 3), dtype=int)

for i in range(len(res)):
    cluster = res[i]
    for j in range(len(cluster)):
        y_index = int(cluster[j] % y)
        x_index = int((cluster[j] - y_index) / y)
        clustered_img[x_index][y_index] = random_color[i]

plt.imshow(clustered_img)

