# -*- coding: utf-8 -*-

import pickle as pkl
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from acell import preprocess_data, load_assist_data
from tgcn import tgcnCell

from visualization import plot_result, plot_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

import argparse
import time

time_start = time.time()

###### Settings ######
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--training_epoch', type=int, default=3000, help='Number of epochs to train.')
parser.add_argument('--gru_units', type=int, default=128, help='Hidden units of gru.')
parser.add_argument('--seq_len', type=int, default=10, help='Time length of inputs.')
parser.add_argument('--pre_len', type=int, default=3, help='Time length of prediction.')
parser.add_argument('--train_rate', type=float, default=0.8, help='Rate of training set.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
parser.add_argument('--dataset', type=str, default='sz', help='Dataset name.')
parser.add_argument('--model_name', type=str, default='tgcn', help='Model name.')
parser.add_argument('--scheme', type=int, default=1, help='Scheme.')
parser.add_argument('--noise_name', type=str, default='None', help='Noise type (None, Gauss, or Poisson).')
parser.add_argument('--noise_param', type=int, default=0, help='Parameter for noise.')
parser.add_argument('--dim', type=int, default=20, help='Feature dimension.')
FLAGS = parser.parse_args()

# 示例：打印参数
print("Learning rate:", FLAGS.learning_rate)
print("Training epochs:", FLAGS.training_epoch)
print("GRU units:", FLAGS.gru_units)
print("Sequence length:", FLAGS.seq_len)
print("Prediction length:", FLAGS.pre_len)
print("Training rate:", FLAGS.train_rate)
print("Batch size:", FLAGS.batch_size)
print("Dataset:", FLAGS.dataset)
print("Model name:", FLAGS.model_name)
print("Scheme:", FLAGS.scheme)
print("Noise name:", FLAGS.noise_name)
print("Noise parameter:", FLAGS.noise_param)

model_name = FLAGS.model_name
noise_name = FLAGS.noise_name
data_name = FLAGS.dataset
train_rate = FLAGS.train_rate
seq_len = FLAGS.seq_len
output_dim = pre_len = FLAGS.pre_len
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units
dim = FLAGS.dim
scheme = FLAGS.scheme
PG = FLAGS.noise_param

###### load data ######
if data_name == 'sz':
    data, adj = load_assist_data('sz')

### Perturbation Analysis
def MaxMinNormalization(x, Max, Min):
    x = (x - Min) / (Max - Min)
    return x

if noise_name == 'Gauss':
    Gauss = np.random.normal(0, PG, size=data.shape)
    noise_Gauss = MaxMinNormalization(Gauss, np.max(Gauss), np.min(Gauss))
    data = data + noise_Gauss
elif noise_name == 'Possion':
    Possion = np.random.poisson(PG, size=data.shape)
    noise_Possion = MaxMinNormalization(Possion, np.max(Possion), np.min(Possion))
    data = data + noise_Possion
else:
    data = data

time_len = data.shape[0]
num_nodes = data.shape[1]
data1 = np.mat(data, dtype=np.float32)

#### normalization
max_value = np.max(data1)
data1 = data1 / max_value

data1 = pd.DataFrame(data1)
data1.columns = data.columns
if model_name == 'ast-gcn':
    if scheme == 1:
        name = 'add poi dim'
    elif scheme == 2:
        name = 'add weather dim'
    else:
        name = 'add poi + weather dim'
else:
    name = 'tgcn'

print('model:', model_name)
print('scheme:', name)
print('noise_name:', noise_name)
print('noise_param:', PG)

trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len, model_name, scheme)

totalbatch = int(trainX.shape[0] / batch_size)
training_data_count = len(trainX)

def TGCN(_X, _weights, _biases):
    cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
    cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.compat.v1.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i, shape=[-1, num_nodes, gru_units])
        o = tf.reshape(o, shape=[-1, gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output, shape=[-1, num_nodes, pre_len])
    output = tf.transpose(output, perm=[0, 2, 1])
    output = tf.reshape(output, shape=[-1, num_nodes])
    return output, m, states

###### placeholders ######
if model_name == 'ast-gcn':
    if scheme == 1:
        inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, seq_len + 1, num_nodes])
    elif scheme == 2:
        inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, seq_len * 2 + pre_len, num_nodes])
    else:
        inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, seq_len * 2 + pre_len + 1, num_nodes])
else:
    inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, seq_len, num_nodes])

labels = tf.compat.v1.placeholder(tf.float32, shape=[None, pre_len, num_nodes])

# Graph weights
weights = {
    'out': tf.Variable(tf.random.normal([gru_units, pre_len], mean=1.0), name='weight_o')}
biases = {
    'out': tf.Variable(tf.random.normal([pre_len]), name='bias_o')}

pred, ttts, ttto = TGCN(inputs, weights, biases)

y_pred = pred

###### optimizer ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1, num_nodes])
## loss
print('y_pred_shape:', y_pred.shape)
print('label_shape:', label.shape)
loss = tf.reduce_mean(tf.compat.v1.losses.mean_squared_error(y_pred, label) + Lreg)
## rmse
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred - label)))
optimizer = tf.compat.v1.train.AdamOptimizer(lr).minimize(loss)

###### Initialize session ######
variables = tf.compat.v1.global_variables()
saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
sess.run(tf.compat.v1.global_variables_initializer())

out = 'out/%s_%s' % (model_name, noise_name)
path1 = '%s_%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r_scheme%r_PG%r' % (
    model_name, name, data_name, lr, batch_size, gru_units, seq_len, pre_len, training_epoch, scheme, PG)
path = os.path.join(out, path1)
if not os.path.exists(path):
    os.makedirs(path)

###### evaluation ######
def evaluation(a, b):
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a - b, 'fro') / la.norm(a, 'fro')
    r2 = 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    var = 1 - (np.var(a - b)) / np.var(a)
    return rmse, mae, 1 - F_norm, r2, var

x_axe, batch_loss, batch_rmse, batch_pred = [], [], [], []
test_loss, test_rmse, test_mae, test_acc, test_r2, test_var, test_pred = [], [], [], [], [], [], []

for epoch in range(training_epoch):
    for m in range(totalbatch):
        mini_batch = trainX[m * batch_size: (m + 1) * batch_size]
        mini_label = trainY[m * batch_size: (m + 1) * batch_size]
        _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, y_pred],
                                                 feed_dict={inputs: mini_batch, labels: mini_label})
        batch_loss.append(loss1)
        batch_rmse.append(rmse1 * max_value)

    # Test completely at every epoch
    loss2, rmse2, test_output = sess.run([loss, error, y_pred],
                                         feed_dict={inputs: testX, labels: testY})

    testoutput = np.abs(test_output)
    test_label = np.reshape(testY, [-1, num_nodes])
    rmse, mae, acc, r2_score, var_score = evaluation(test_label, testoutput)
    test_label1 = test_label * max_value
    test_output1 = testoutput * max_value
    test_loss.append(loss2)
    test_rmse.append(rmse * max_value)
    test_mae.append(mae * max_value)
    test_acc.append(acc)
    test_r2.append(r2_score)
    test_var.append(var_score)
    test_pred.append(test_output1)

    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(batch_rmse[-1]),
          'test_loss:{:.4}'.format(loss2),
          'test_rmse:{:.4}'.format(rmse),
          'test_acc:{:.4}'.format(acc))

    if (epoch % 500 == 0):
        saver.save(sess, path + '/model_100/ASTGCN_pre_%r' % epoch, global_step=epoch)

time_end = time.time()
print(time_end - time_start, 's')
b = int(len(batch_rmse) / totalbatch)
batch_rmse1 = [i for i in batch_rmse]
train_rmse = [sum(batch_rmse1[i * totalbatch : (i + 1) * totalbatch]) / totalbatch for i in range(b)]
batch_loss1 = [i for i in batch_loss]
train_loss = [sum(batch_loss1[i * totalbatch : (i + 1) * totalbatch]) / totalbatch for i in range(b)]

index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
var = pd.DataFrame(test_result)
var.to_csv(path + '/test_result.csv', index=False, header=False)
plot_result(test_result, test_label1, path)
plot_error(train_rmse, train_loss, test_rmse, test_acc, test_mae, path)
evalution = []
evalution.append(np.min(test_rmse))
evalution.append(test_mae[index])
evalution.append(test_acc[index])
evalution.append(test_r2[index])
evalution.append(test_var[index])
evalution = pd.DataFrame(evalution)
evalution.to_csv(path + '/evalution.csv', index=False, header=None)
print('model_name:', model_name)
print('scheme:', scheme)
print('name:', name)
print('noise_name:', noise_name)
print('PG:', PG)
print('min_rmse:%r' % (np.min(test_rmse)),
      'min_mae:%r' % (test_mae[index]),
      'max_acc:%r' % (test_acc[index]),
      'r2:%r' % (test_r2[index]),
      'var:%r' % test_var[index])
