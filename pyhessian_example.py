"""
    hessian_estimator_example.py - Hessian Matrix Estimator Example
 
    Copyright (c) 1996-2016 by Geir K. Nilsen (geir.kjetil.nilsen@gmail.com)
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
import pytictoc
TicToc = pytictoc.TicToc()

# Model architecture; number of neurons, layer-wise.
# e.g. Multinomial Logistic Regression
layers = [64, 128]

# Initialize random dummy weights & biases (no training, just for example)
W = tf.Variable(tf.random.normal((layers[0], layers[1])), 'float32')
b = tf.Variable(tf.random.normal((layers[1], )), 'float32')

# Stack weights layer-wise first, then biases layer-wise after
params = [W, b]

# Input-output variables
X = tf.placeholder(dtype='float32', shape=(None, layers[0]), name="X-data")
y = tf.placeholder(dtype='float32', shape=(None, layers[1]), name="y-data")

# Model function
def model_fun(X, params):
    return tf.add(tf.matmul(X, params[0]), params[1])
        
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
batch_size = 100

# Initialize HessianEstimator object
hest = HessianEstimator(layers, cost_fun, cost, model_fun, params, 
                        X, y, batch_size)

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
X_train = np.random.normal(size=(N, layers[0]))
y_train = np.random.normal(size=(N, layers[1]))

# Evaluate first column of full Hessian matrix
Hv = sess.run(Hv_op, feed_dict={X:X_train, y:y_train})

# Evaluate first column of mini-batch Hessian matrix
Hv = sess.run(Hv_op, feed_dict={X:X_train[:batch_size], 
                                y:y_train[:batch_size]})

# Evaluate first column of single example Hessian matrix
Hv = sess.run(Hv_op, feed_dict={X:[X_train[0]], 
                                y:[y_train[0]]})

# Evaluate full Hessian matrix
H = sess.run(H_op, feed_dict={X:X_train, y:y_train})

# Evaluate mini-batch Hessian matrix
N = 64000
batch_size=N
X_train = np.random.normal(size=(N, layers[0]))
y_train = np.random.normal(size=(N, layers[1]))

TicToc.tic()
H = sess.run(H_op, feed_dict={X:X_train[:batch_size], 
                              y:y_train[:batch_size]})
TicToc.toc()

# Evaluate single example Hessian matrix
H = sess.run(H_op, feed_dict={X:[X_train[0]], 
                              y:[y_train[0]]})

# Evaluate full OPG matrix
G = np.zeros(hest.P, hest.P)
B = int(N/batch_size)
TicToc.tic()
for b in range(B):
    G = G + sess.run(G_op, feed_dict={X:X_train[b*batch_size: \
                                                (b+1)*batch_size], 
                                      y:y_train[b*batch_size: \
                                                (b+1)*batch_size]})
G = G / B
TicToc.toc()

# Evaluate mini-batch OPG matrix
G = sess.run(G_op, feed_dict={X:X_train[:batch_size], 
                              y:y_train[:batch_size]})

# Evaluate single example Hessian OPG approximation matrix (must re-init HessianEstimator 
# with batch_size=1)
batch_size = 1 
hest = HessianEstimator(layers, cost_fun, cost, model_fun, params, 
                        X, y, batch_size)
G_op = hest.get_G_op()
G = sess.run(G_op, feed_dict={X:[X_train[0]], 
                              y:[y_train[0]]})
