import tensorflow as tf
import numpy as np

input_map = np.random.random((1, 10, 10, 3))
input_tensor = tf.constant(input_map, dtype=tf.float32)
"""2(10-1)+3-1"""
# 使用Conv2DTranspose层创建output
conv2d_transpose_layer = tf.keras.layers.Conv2DTranspose(1, 3, 2, padding='same')

output = conv2d_transpose_layer(input_tensor)

# 打印output的形状
print("Output Shape:", output.shape)
