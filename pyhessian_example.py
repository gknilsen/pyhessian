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

import tensorflow.compat.v1 as tf
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.disable_v2_behavior()
import numpy as np
from pyhessian import HessianEstimator

# Model architecture; number of neurons, layer-wise.
# e.g. feed-forward neural network
T1, T2, T3, T4 = 8, 4, 4, 2

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
# Batch size for Hessian estimator
batch_size_H = 1000

# # Initialize HessianEstimator object
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
Hv = sess.run(Hv_op, feed_dict={X:X_train[:batch_size_H], 
                                y:y_train[:batch_size_H]})

# Evaluate first column of single example Hessian matrix
Hv = sess.run(Hv_op, feed_dict={X:[X_train[0]], 
                                y:[y_train[0]]})

# Evaluate full Hessian matrix
H = np.zeros((hest.P, hest.P), dtype='float32')
B = int(N/batch_size_H)
for b in range(B):
    H = H + sess.run(H_op, feed_dict={X:X_train[b*batch_size_H: \
                                                (b+1)*batch_size_H], 
                                      y:y_train[b*batch_size_H: \
                                                (b+1)*batch_size_H]})
H = H / B

# Evaluate mini-batch Hessian matrix
H = sess.run(H_op, feed_dict={X:X_train[:batch_size_H], 
                              y:y_train[:batch_size_H]})

# Evaluate single example Hessian matrix
H = sess.run(H_op, feed_dict={X:[X_train[0]], 
                              y:[y_train[0]]})

# Evaluate full OPG matrix
G = np.zeros((hest.P, hest.P), dtype='float32')
B = int(N/batch_size_G)
for b in range(B):
    G = G + sess.run(G_op, feed_dict={X:X_train[b*batch_size_G: \
                                                (b+1)*batch_size_G], 
                                      y:y_train[b*batch_size_G: \
                                                (b+1)*batch_size_G]})
G = G / B

# Evaluate mini-batch OPG matrix
G = sess.run(G_op, feed_dict={X:X_train[:batch_size_G], 
                              y:y_train[:batch_size_G]})

# Evaluate single example Hessian OPG approximation matrix 
# (must re-init HessianEstimator with batch_size=1)
batch_size_G = 1 
hest = HessianEstimator(cost_fun, cost, model_fun, params, 
                        X, y, batch_size_G)
G_op = hest.get_G_op()
G = sess.run(G_op, feed_dict={X:[X_train[0]], 
                              y:[y_train[0]]})

# For very large models where the full Hessian matrix is too large to fit in memory, 
# an alternative is to use approximate eigendecompositons:

# Compute eigendecomposition of H 
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh
K = 10
hest = HessianEstimator(cost_fun, cost, model_fun, params, 
                        X, y, batch_size_G)
_v = tf.placeholder(shape=(hest.P,), dtype='float32')
Hv_op = hest.get_Hv_op(_v)
def Hv(v):
    B = int(N/batch_size_H)
    Hv = np.zeros((hest.P))
    Bs = batch_size_H
    for b in range(B):
        Hv = Hv + sess.run(Hv_op, 
                           feed_dict={X:X_train[b*Bs: \
                                                (b+1)*Bs], 
                                      y:y_train[b*Bs: \
                                                (b+1)*Bs], 
                                      _v:np.squeeze(v)})
    Hv = Hv / B
    return Hv
H = LinearOperator((hest.P, hest.P), matvec=Hv, dtype='float32')
L_H, Q_H = eigsh(H, k=K, which='LA')


# Compute eigendecomposition of G
from sklearn.decomposition import IncrementalPCA
K = 10
hest = HessianEstimator(cost_fun, cost, model_fun, params, 
                        X, y, batch_size_G)
_N = int(np.ceil(K / batch_size_G))

if N % _N != 0:
    raise Exception('N must be divisible by K/batch_size_G!')

ipca = IncrementalPCA(n_components=K, batch_size=batch_size_G*_N, 
                      copy=False)

J_op = hest.get_J_op()

J = np.zeros((batch_size_G*_N, hest.P), dtype='float32')
B = int(N/batch_size_G)
Bs = batch_size_G
for b in range(B):
    J[Bs*(b%_N):Bs*(b%_N+1),:] = sess.run(J_op, 
                                          feed_dict={X: X_train[b*Bs:\
                                                                (b+1)*Bs], 
                                                     y: y_train[b*Bs:\
                                                                (b+1)*Bs]})
    if (b+1) % _N == 0:
        ipca.partial_fit(J)

L_G, Q_G = np.float32(ipca.singular_values_**2 / N),\
           np.float32(ipca.components_.T)

