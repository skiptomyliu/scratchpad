

import tensorflow as tf
import numpy as np


# x_data = np.random.rand(100).astype(np.float32)
# y_data = x_data * 0.1 + 0.3

# W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# b = tf.Variable(tf.zeros([1]))
# y = W*x_data + b

# loss = tf.reduce_mean(tf.square(y-y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train = optimizer.minimize(loss)

class TwoLayerNet(object):

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
        self.hidden_dim=hidden_dim
        self.weight_scale = weight_scale

    def loss(self, X, y):
        weight_scale=1e-3
        input_dim=3*32*32
        hidden_dim=100
        W = weight_scale * np.random.randn(input_dim, hidden_dim)
        
        X = np.reshape(X, (X.shape[0], -1))

        # where is W?
        L1_scores = tf.matmul(W, x_data)
        # tf.nn.relu()
        tf.nn.softmax(tf.matmul(W,x_data))

        init = tf.initialize_all_variables()

        # Launch graph
        sess = tf.Session()
        sess.run(init)

# for step in range(100):
#     sess.run(train)
#     if step%20 == 0:
#         print(step, sess.run(W), sess.run(b), sess.run(loss))


