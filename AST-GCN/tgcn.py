# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import AbstractRNNCell
from utils import calculate_laplacian

class tgcnCell(AbstractRNNCell):
    def __init__(self, num_units, adj, num_nodes, input_size=None, act=tf.nn.tanh, **kwargs):
        super().__init__(**kwargs)
        self._act = act
        self._nodes = num_nodes
        self._units = num_units
        self._adj = [calculate_laplacian(adj)]

    def zero_state(self, batch_size, dtype):
        return [tf.zeros((batch_size, self._nodes * self._units), dtype=dtype)]

    @property
    def state_size(self):
        return self._nodes * self._units

    @property
    def output_size(self):
        return self._units

    def call(self, inputs, states):
        state = tf.cond(
            tf.greater(tf.size(states), 0),
            lambda: states[0],
            lambda: tf.zeros((tf.shape(inputs)[0], self.state_size), dtype=tf.float32)
        )

        with tf.name_scope("tgcn"):
            with tf.name_scope("gates"):
                value = tf.nn.sigmoid(self._gc(inputs, state, 2 * self._units, bias=1.0))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
            with tf.name_scope("candidate"):
                r_state = r * state
                c = self._act(self._gc(inputs, r_state, self._units))
            new_h = u * state + (1 - u) * c
        return new_h, [new_h]

    def _gc(self, inputs, state, output_size, bias=0.0):
        inputs = tf.expand_dims(inputs, 2)
        state = tf.reshape(state, (-1, self._nodes, self._units))
        x_s = tf.concat([inputs, state], axis=2)
        input_size = x_s.shape[2]

        x0 = tf.transpose(x_s, perm=[1, 2, 0])
        x0 = tf.reshape(x0, shape=[self._nodes, -1])

        scope = tf.compat.v1.get_variable_scope()
        with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
            for m in self._adj:
                x1 = tf.compat.v1.sparse_tensor_dense_matmul(m, x0)
            x = tf.reshape(x1, shape=[self._nodes, input_size, -1])
            x = tf.transpose(x, perm=[2, 0, 1])
            x = tf.reshape(x, shape=[-1, input_size])

            # 关键修改：根据 output_size 动态生成变量名
            weights = tf.compat.v1.get_variable(
                f'weights_{output_size}',  # 添加 output_size 后缀确保唯一性
                [input_size, output_size],
                initializer=tf.keras.initializers.GlorotNormal(),
                dtype=tf.float32
            )
            biases = tf.compat.v1.get_variable(
                f'biases_{output_size}',  # 添加 output_size 后缀确保唯一性
                [output_size],
                initializer=tf.constant_initializer(bias),
                dtype=tf.float32
            )

            x = tf.matmul(x, weights)
            x = tf.nn.bias_add(x, biases)
            x = tf.reshape(x, shape=[-1, self._nodes, output_size])
            x = tf.reshape(x, shape=[-1, self._nodes * output_size])
        return x


# # -*- coding: utf-8 -*-
#
# import tensorflow as tf
# from tensorflow.keras.layers import AbstractRNNCell
# from utils import calculate_laplacian
# import numpy as np
# import pandas as pd
#
#
# class tgcnCell(AbstractRNNCell):
#     def __init__(self, num_units, adj, num_nodes, input_size=None, act=tf.nn.tanh, **kwargs):
#         super().__init__(**kwargs)
#         self._act = act
#         self._nodes = num_nodes
#         self._units = num_units
#         self._adj = [calculate_laplacian(adj)]
#
#     def zero_state(self, batch_size, dtype):
#         # 返回一个符合state_size形状的初始状态
#         return [tf.zeros((batch_size, self.state_size), dtype=dtype)]
#
#     @property
#     def state_size(self):
#         return self._nodes * self._units  # 返回正确的状态尺寸
#
#     @property
#     def output_size(self):
#         return self._units
#
#     def call(self, inputs, states):
#         state = tf.cond(
#             tf.greater(tf.size(states), 0),
#             lambda: states[0],
#             lambda: tf.zeros((tf.shape(inputs)[0], self.state_size), dtype=tf.float32)
#         )
#
#         with tf.name_scope("tgcn"):
#             with tf.name_scope("gates"):
#                 value = tf.nn.sigmoid(self._gc(inputs, state, 2 * self._units, bias=1.0))
#                 r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
#             with tf.name_scope("candidate"):
#                 r_state = r * state
#                 c = self._act(self._gc(inputs, r_state, self._units))
#             new_h = u * state + (1 - u) * c
#         return new_h, [new_h]
# # class tgcnCell(AbstractRNNCell):
# #     """Temporal Graph Convolutional Network """
# #
# #     def __init__(self, num_units, adj, num_nodes, input_size=None, act=tf.nn.tanh, **kwargs):
# #         super().__init__(**kwargs)
# #         self._act = act
# #         self._nodes = num_nodes
# #         self._units = num_units
# #         self._adj = [calculate_laplacian(adj)]
# #
# #     def zero_state(self, batch_size, dtype):
# #         return [tf.zeros((batch_size, self._nodes * self._units), dtype=dtype)]
# #
# #     @property
# #     def state_size(self):
# #         return (self._nodes * self._units,)  # 返回元组
# #
# #     @property
# #     def output_size(self):
# #         return self._units
# #
# #     def call(self, inputs, states):
# #         state = tf.cond(
# #             tf.greater(tf.size(states), 0),  # 检查states是否非空
# #             lambda: states[0],  # 如果非空，取第一个状态
# #             lambda: tf.zeros((tf.shape(inputs)[0], self.state_size), dtype=tf.float32)
# #         )# 否则创建零状态
# #         # 输入形状处理
# #        # state = states[0] if states else tf.zeros((tf.shape(inputs)[0], self.state_size))
# #
# #         with tf.name_scope("tgcn"):
# #             with tf.name_scope("gates"):
# #                 value = tf.nn.sigmoid(
# #                     self._gc(inputs, state, 2 * self._units, bias=1.0))
# #                 r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
# #             with tf.name_scope("candidate"):
# #                 r_state = r * state
# #                 c = self._act(self._gc(inputs, r_state, self._units))
# #             new_h = u * state + (1 - u) * c
# #         return new_h, [new_h]
# #
#     def _gc(self, inputs, state, output_size, bias=0.0):
#         ## inputs:(-1,num_nodes)
#         inputs = tf.expand_dims(inputs, 2)
#         #        print('inputs_shape:',inputs.shape)
#         ## state:(batch,num_node,gru_units)
#         state = tf.reshape(state, (-1, self._nodes, self._units))
#         #        print('state_shape:',state.shape)
#         ## concat
#         x_s = tf.concat([inputs, state], axis=2)
#         #        print('x_s_shape:',x_s.shape)
#
#         #        kgembedding = np.array(pd.read_csv(r'/DHH/sz_gcn/sz_data/sz_poi_transR_embedding20.csv',header=None))
#         #        kgeMatrix = np.repeat(kgembedding[np.newaxis, :, :], self._units, axis=0)
#         #        kgeMatrix = tf.reshape(tf.constant(kgeMatrix, dtype=tf.float32), (self._units, -1))
#         #        kgMatrix = tf.reshape(kgeMatrix,(-1,self._nodes, 20))
#         #
#         #        ## inputs:(-1,num_nodes)
#         #        inputs = tf.expand_dims(inputs, 2)
#         #        ## state:(batch,num_node,gru_units)
#         #        state = tf.reshape(state, (-1, self._nodes, self._units))
#         #        ## concat
#         #        print('kgMatrix_shape:',kgMatrix.shape)
#         #        print('inputs_shape:',inputs.shape)
#         #        print('state_shape:',state.shape)
#         #        kg_x = tf.concat([inputs, kgMatrix],axis = 2)
#         #        print('kg_x_shape:',kg_x.shape)
#         #        x_s = tf.concat([kg_x, state], axis=2)
#         input_size = x_s.get_shape()[2].value
#         ## (num_node,input_size,-1)
#         x0 = tf.transpose(x_s, perm=[1, 2, 0])
#         x0 = tf.reshape(x0, shape=[self._nodes, -1])
#
#         scope =tf.compat.v1.get_variable_scope()
#         with tf.compat.v1.variable_scope(scope):
#             for m in self._adj:
#                 x1 = tf.compat.v1.sparse_tensor_dense_matmul(m, x0)
#             #                print(x1)
#             x = tf.compat.v1.reshape(x1, shape=[self._nodes, input_size, -1])
#             x = tf.compat.v1.transpose(x, perm=[2, 0, 1])
#             x = tf.compat.v1.reshape(x, shape=[-1, input_size])
#             weights = tf.compat.v1.get_variable(
#                 'weights', [input_size, output_size], initializer=tf.keras.initializers.GlorotNormal())
#             x = tf.matmul(x, weights)  # (batch_size * self._nodes, output_size)
#             # biases = tf.compat.v1.get_variable(
#             #     "biases", [output_size], initializer=tf.constant_initializer(bias, dtype=tf.float32))
#             #
#             biases = tf.compat.v1.get_variable(
#                 'biases', [output_size],
#                 initializer=tf.constant_initializer(bias),
#                 dtype=tf.float32
#             )
#
#             x = tf.nn.bias_add(x, biases)
#             x = tf.reshape(x, shape=[-1, self._nodes, output_size])
#             x = tf.reshape(x, shape=[-1, self._nodes * output_size])
#         return x