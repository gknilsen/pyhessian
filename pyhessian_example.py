"""
    pyhessian_example.py - pyhessian Usage Example for a Feed-Forward
    Neural Network model.
     
    Copyright (c) 2018-2019 by Geir K. Nilsen (geir.kjetil.nilsen@gmail.com)
    and the University of Bergen.
 
    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.
 
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import tensorflow as tf
import numpy as np
from pyhessian import HessianEstimator

# Model architecture; number of neurons, layer-wise.
# e.g. feed-forward neural network
T1, T2, T3, T4 = 128, 64, 64, 32

# Initialize random dummy weights & biases (no training, just for example)
W1 = tf.Variable(tf.random.normal((T1, T2)), 'float32')
W2 = tf.Variable(tf.random.normal((T2, T3)), 'float32')
W3 = tf.Variable(tf.random.normal((T3, T4)), 'float32')

b2 = tf.Variable(tf.random.normal((T2, )), 'float32')
b3 = tf.Variable(tf.random.normal((T3, )), 'float32')
b4 = tf.Variable(tf.random.normal((T4, )), 'float32')

# Stack weights and biases layer-wise
params = [W1, b2, W2, b3, W3, b4]

# Input-output variables
X = tf.placeholder(dtype='float32', shape=(None, T1))
y = tf.placeholder(dtype='float32', shape=(None, T4))

# Model function
def model_fun(X, params):
    l_2 = tf.nn.softplus(tf.add(tf.matmul(X, params[0]), params[1]))
    l_3 = tf.nn.softplus(tf.add(tf.matmul(l_2, params[2]), params[3]))
    l_4 = tf.add(tf.matmul(l_3, params[4]), params[5])
    return l_4
        
# Model output (logits)
yhat_logits = model_fun(X, params)
# Model output (softmax)
yhat = tf.nn.softmax(yhat_logits)

# Cost function
def cost_fun(y, yhat_logits, params):
    return tf.losses.softmax_cross_entropy(y, yhat_logits)

# Cost output
cost = cost_fun(y, yhat_logits, params)

# Batch size for OPG estimator
batch_size_G = 100

# Initialize HessianEstimator object
hest = HessianEstimator(cost_fun, cost, model_fun, params, 
                        X, y, batch_size_G)

# First Hessian column op
Hv_op = hest.get_Hv_op(tf.eye(hest.P, 1))

# Full Hessian op
H_op = hest.get_H_op()

# Full OPG op
G_op = hest.get_G_op()

# Init graph session
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

# Define dummy training data, N examples
N = 1000
X_train = np.random.normal(size=(N, T1))
y_train = np.random.normal(size=(N, T4))

# Evaluate first column of full Hessian matrix
Hv = sess.run(Hv_op, feed_dict={X:X_train, y:y_train})

# Evaluate first column of mini-batch Hessian matrix
Hv = sess.run(Hv_op, feed_dict={X:X_train[:batch_size], 
                                y:y_train[:batch_size]})

# Evaluate first column of single example Hessian matrix
Hv = sess.run(Hv_op, feed_dict={X:[X_train[0]], 
                                y:[y_train[0]]})

# Evaluate full Hessian matrix
H = np.zeros((hest.P, hest.P), dtype='float32')
B = int(N/batch_size)
for b in range(B):
    H = H + sess.run(H_op, feed_dict={X:X_train[b*batch_size: \
                                                (b+1)*batch_size], 
                                      y:y_train[b*batch_size: \
                                                (b+1)*batch_size]})
H = H / B

# Evaluate mini-batch Hessian matrix
H = sess.run(H_op, feed_dict={X:X_train[:batch_size], 
                              y:y_train[:batch_size]})

# Evaluate single example Hessian matrix
H = sess.run(H_op, feed_dict={X:[X_train[0]], 
                              y:[y_train[0]]})

# Evaluate full OPG matrix
G = np.zeros((hest.P, hest.P), dtype='float32')
B = int(N/batch_size)
for b in range(B):
    G = G + sess.run(G_op, feed_dict={X:X_train[b*batch_size: \
                                                (b+1)*batch_size], 
                                      y:y_train[b*batch_size: \
                                                (b+1)*batch_size]})
G = G / B

# Evaluate mini-batch OPG matrix
G = sess.run(G_op, feed_dict={X:X_train[:batch_size], 
                              y:y_train[:batch_size]})

# Evaluate single example Hessian OPG approximation matrix 
# (must re-init HessianEstimator with batch_size=1)
batch_size = 1 
hest = HessianEstimator(cost_fun, cost, model_fun, params, 
                        X, y, batch_size)
G_op = hest.get_G_op()
G = sess.run(G_op, feed_dict={X:[X_train[0]], 
                              y:[y_train[0]]})
