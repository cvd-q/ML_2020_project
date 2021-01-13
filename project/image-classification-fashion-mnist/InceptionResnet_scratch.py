from tensorflow.python.keras import backend
from tensorflow.python.keras.layers import VersionAwareLayers
import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model, load_model
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
  """Utility function to apply conv + BN.
  Arguments:
    x: input tensor.
    filters: filters in `Conv2D`.
    kernel_size: kernel size as in `Conv2D`.
    strides: strides in `Conv2D`.
    padding: padding mode in `Conv2D`.
    activation: activation in `Conv2D`.
    use_bias: whether to use a bias in `Conv2D`.
    name: name of the ops; will become `name + '_ac'` for the activation
        and `name + '_bn'` for the batch norm layer.
  Returns:
    Output tensor after applying `Conv2D` and `BatchNormalization`.
  """
  x = layers.Conv2D(
      filters,
      kernel_size,
      strides=strides,
      padding=padding,
      use_bias=use_bias,
      name=name)(x)
  if not use_bias:
    bn_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    bn_name = None if name is None else name + '_bn'
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
  if activation is not None:
    ac_name = None if name is None else name + '_ac'
    x = layers.Activation(activation, name=ac_name)(x)
  return x

def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds an Inception-ResNet block.
      This function builds 3 types of Inception-ResNet blocks mentioned
      in the paper, controlled by the `block_type` argument (which is the
      block name used in the official TF-slim implementation):
      - Inception-ResNet-A: `block_type='block35'`
      - Inception-ResNet-B: `block_type='block17'`
      - Inception-ResNet-C: `block_type='block8'`
      Arguments:
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of passing
          `x` through an inception module) before adding them to the shortcut
          branch. Let `r` be the output from the residual branch, the output of this
          block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines the network
          structure in the residual branch.
        block_idx: an `int` used for generating layer names. The Inception-ResNet
          blocks are repeated many times in this network. We use `block_idx` to
          identify each of the repetitions. For example, the first
          Inception-ResNet-A block will have `block_type='block35', block_idx=0`,
          and the layer names will have a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block (see
          [activations](../activations.md)). When `activation=None`, no activation
          is applied
          (i.e., "linear" activation: `a(x) = x`).
      Returns:
          Output tensor for the block.
      Raises:
        ValueError: if `block_type` is not one of `'block35'`,
          `'block17'` or `'block8'`.
    """
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    mixed = layers.Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_bn(
      mixed,
      backend.int_shape(x)[channel_axis],
      1,
      activation=None,
      use_bias=True,
      name=block_name + '_conv')

    x = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
        output_shape=backend.int_shape(x)[1:],
        arguments={'scale': scale},
        name=block_name)([x, up])
    if activation is not None:
        x = layers.Activation(activation, name=block_name + '_ac')(x)
    return x

class InceptionResNetV2Fashion(Model):
    """
    Modified InceptionResNetV2 (tf.keras.application), it's smaller model in order to fit low resolution images (28 x 28).

    Arguments:
        input_shape: a integer tuple representing the image dimension (it can be 4-dim tuple with batch size at first pos.
        or 3-dim tuple with channel number at the end).

        (COMPLETE!)

    Returns:
          A Model class object with embedded layers.
    """

    def __init__(self, input_shape, n_class):
        self.N_CLASS = tf.constant(n_class)

        inputs = Input(input_shape)
        # Stem block: 24 x 24 x 192
        x = conv2d_bn(inputs, 16, 3)
        x = conv2d_bn(x, 16, 3)
        x = conv2d_bn(x, 32, 3)
        x = layers.MaxPooling2D(3, strides=1)(x)
        x = conv2d_bn(x, 40, 1)
        x = conv2d_bn(x, 96, 3)
        x = layers.MaxPooling2D(3, strides=1)(x)
        x = layers.Dropout(0.6)(x)

        # 5x block35 (Inception-ResNet-A block): 24 x 24 x 320
        for block_idx in range(1, 6):
            x = inception_resnet_block(
                x, scale=0.17, block_type='block35', block_idx=block_idx)
        # Mixed 6a (Reduction-A block): 24 x 24 x 1088
        branch_0 = conv2d_bn(x, 384/2, 3, strides=2, padding='valid')
        branch_1 = conv2d_bn(x, 256/2, 1)
        branch_1 = conv2d_bn(branch_1, 256/2, 3)
        branch_1 = conv2d_bn(branch_1, 384/2, 3, strides=2, padding='valid')
        branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
        branches = [branch_0, branch_1, branch_pool]
        channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
        x = layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)
        x = layers.Dropout(0.6)(x)

        # 10x block17 (Inception-ResNet-B block): 24 x 24 x 1088
        for block_idx in range(1, 11):
            x = inception_resnet_block(
                x, scale=0.1, block_type='block17', block_idx=block_idx)

        # Mixed 7a (Reduction-B block): 11 x 11 x 2080
        branch_0 = conv2d_bn(x, 256/2, 1)
        branch_0 = conv2d_bn(branch_0, 384/2, 3, strides=2, padding='valid')
        branch_1 = conv2d_bn(x, 256/2, 1)
        branch_1 = conv2d_bn(branch_1, 288/2, 3, strides=2, padding='valid')
        branch_2 = conv2d_bn(x, 256/2, 1)
        branch_2 = conv2d_bn(branch_2, 288/2, 3)
        branch_2 = conv2d_bn(branch_2, 320/2, 3, strides=2, padding='valid')
        branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = layers.Concatenate(axis=channel_axis, name='mixed_7a')(branches)
        x = layers.Dropout(0.6)(x)

        # 5x block8 (Inception-ResNet-C block): 11 x 11 x 2080
        for block_idx in range(1, 5):
            x = inception_resnet_block(
                x, scale=0.2, block_type='block8', block_idx=block_idx)
        x = inception_resnet_block(
            x, scale=1., activation=None, block_type='block8', block_idx=5)
        x = layers.Dropout(0.6)(x)

        # Final convolution block: 5 x 5 x 1536
        x = conv2d_bn(x, 1536/2, 1, name='conv_7b')
        x = layers.Dropout(0.6)(x)

        # Vectorize using average pooling
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.6)(x)
        x = layers.Flatten()(x)

        x = tf.keras.layers.Dense(500, activation='relu')(x)
        x = layers.Dropout(0.6)(x)
        x = tf.keras.layers.Dense(int(500/10), activation='relu')(x)
        x = layers.Dropout(0.6)(x)
        x = tf.keras.layers.Dense(self.N_CLASS, activation='softmax')(x)  # N_CLASSE = 10

        super(InceptionResNetV2Fashion, self).__init__(inputs = inputs, outputs = x)
'''    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
'''

model = InceptionResNetV2Fashion((28,28,1), 10)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                                            0.01,
                                                                            decay_steps=300,
                                                                            decay_rate=0.90,
                                                                            staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer)