from minisom import MiniSom
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

# 读取图片
img = plt.imread('./data/flower.jpg')

# 重新构建像素矩阵
pixels = np.reshape(img, (img.shape[0]*img.shape[1], 3))

# SOM模型
print('training...')
som = MiniSom(3, 3, 3, sigma=1.,
              learning_rate=0.2, neighborhood_function='bubble')
som.random_weights_init(pixels)
starting_weights = som.get_weights().copy()
start = time.perf_counter()
som.train_random(pixels, 500)
end = time.perf_counter()
print('Running time: %s ms' % ((end - start) * 1000))

# 构建新图片
print('quantization...')
qnt = som.quantization(pixels)
print('building new image...')
clustered = np.zeros(img.shape)
for i, q in enumerate(qnt):
    clustered[np.unravel_index(i, dims=(img.shape[0], img.shape[1]))] = q
print('done.')

# 显示结果
plt.figure(figsize=(7, 7))
plt.figure(1)
plt.subplot(221)
plt.title('original')
plt.imshow(img)
plt.subplot(222)
plt.title('result')
plt.imshow(clustered)

plt.subplot(223)
plt.title('initial colors')
plt.imshow(starting_weights, interpolation='none')
plt.subplot(224)
plt.title('learned colors')
plt.imshow(som.get_weights(), interpolation='none')

plt.tight_layout()
plt.savefig('./data/som_color_quantization.png')
plt.show()
