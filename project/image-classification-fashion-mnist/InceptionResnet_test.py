from tensorflow.python.keras import backend
import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model
from tensorboard.plugins.hparams import api as hp
import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tb_imgs_utilities import image_grid, plot_to_image, confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.keras.engine import data_adapter
from sklearn.metrics import roc_auc_score

# CONSTANTS
N_CLASS = 10
enc = OneHotEncoder(handle_unknown='ignore')
cl = np.arange(N_CLASS).reshape((N_CLASS,1))
enc.fit(cl)
log_dir= 'logs/InceptionResnet_test' #'./logs/'
checkpoint_filepath = 'checkpoint_InceptionResnet_test/'#'./checkpoint/'

# PREPARE DATA
DATA_BASE_FOLDER = './' #'/kaggle/input/ml-project-2020-dataset/'
x_train = np.load(os.path.join(DATA_BASE_FOLDER, 'train.npy'))
x_valid = np.load(os.path.join(DATA_BASE_FOLDER, 'validation.npy'))
x_test = np.load(os.path.join(DATA_BASE_FOLDER, 'test.npy'))
y_train = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'train.csv'))['class'].values
y_valid = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'validation.csv'))['class'].values

#only 500 images and test dataset won't be used
# x_train = x_train[0:50]
# x_valid = x_valid[0:50]
# y_train = y_train[0:50]
# y_valid = y_valid[0:50]

y_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
x_train = x_train.reshape(x_train.shape[0], 28, 28) # reconstruct images
x_valid = x_valid.reshape(x_valid.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)
x_train = tf.expand_dims(x_train, axis=-1)
x_valid = tf.expand_dims(x_valid, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)
img_generator = ImageDataGenerator(preprocessing_function=lambda x: x/255., horizontal_flip=True)

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

    def __init__(self, input_shape, n_class,
                 REDUCE_SIZE,
                 DROP_RATE):
        self.N_CLASS = tf.constant(n_class)

        inputs = Input(input_shape)
        # Stem block: 24 x 24
        x = conv2d_bn(inputs, int(32/REDUCE_SIZE), 3)
        x = conv2d_bn(x, int(32/REDUCE_SIZE), 3)
        x = conv2d_bn(x, int(64/REDUCE_SIZE), 3)
        x = layers.MaxPooling2D(3, strides=1)(x)
        x = conv2d_bn(x, int(80/REDUCE_SIZE), 1)
        x = conv2d_bn(x, int(192/REDUCE_SIZE), 3)
        x = layers.MaxPooling2D(3, strides=1)(x)
        x = layers.Dropout(DROP_RATE)(x)

        # 5x block35 (Inception-ResNet-A block): 24 x 24
        for block_idx in range(1, 6):
            x = inception_resnet_block(
                x, scale=0.17, block_type='block35', block_idx=block_idx)
        # Mixed 6a (Reduction-A block): 24 x 24 x 1088
        branch_0 = conv2d_bn(x, int(384/REDUCE_SIZE), 3, strides=2, padding='valid')
        branch_1 = conv2d_bn(x, int(256/REDUCE_SIZE), 1)
        branch_1 = conv2d_bn(branch_1, int(256/REDUCE_SIZE), 3)
        branch_1 = conv2d_bn(branch_1, int(384/REDUCE_SIZE), 3, strides=2, padding='valid')
        branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
        branches = [branch_0, branch_1, branch_pool]
        channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
        x = layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)
        x = layers.Dropout(DROP_RATE)(x)

        # 10x block17 (Inception-ResNet-B block): 24 x 24
        for block_idx in range(1, 11):
            x = inception_resnet_block(
                x, scale=0.1, block_type='block17', block_idx=block_idx)

        # Mixed 7a (Reduction-B block): 11 x 11
        branch_0 = conv2d_bn(x, int(256/REDUCE_SIZE), 1)
        branch_0 = conv2d_bn(branch_0, int(384/REDUCE_SIZE), 3, strides=2, padding='valid')
        branch_1 = conv2d_bn(x, int(256/REDUCE_SIZE), 1)
        branch_1 = conv2d_bn(branch_1, int(288/REDUCE_SIZE), 3, strides=2, padding='valid')
        branch_2 = conv2d_bn(x, int(256/REDUCE_SIZE), 1)
        branch_2 = conv2d_bn(branch_2, int(288/REDUCE_SIZE), 3)
        branch_2 = conv2d_bn(branch_2, int(320/REDUCE_SIZE), 3, strides=2, padding='valid')
        branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = layers.Concatenate(axis=channel_axis, name='mixed_7a')(branches)
        x = layers.Dropout(DROP_RATE)(x)

        # 5x block8 (Inception-ResNet-C block): 11 x 11
        for block_idx in range(1, 5):
            x = inception_resnet_block(
                x, scale=0.2, block_type='block8', block_idx=block_idx)
        x = inception_resnet_block(
            x, scale=1., activation=None, block_type='block8', block_idx=5)
        x = layers.Dropout(DROP_RATE)(x)

        # Final convolution block: 5 x 5
        x = conv2d_bn(x, int(1536/REDUCE_SIZE), 1, name='conv_7b')
        x = layers.Dropout(DROP_RATE)(x)

        # Vectorize using average pooling
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(DROP_RATE)(x)
        x = layers.Flatten()(x)

        x = tf.keras.layers.Dense(500, activation='relu')(x)
        x = layers.Dropout(DROP_RATE)(x)
        x = tf.keras.layers.Dense(50, activation='relu')(x)
        x = layers.Dropout(DROP_RATE)(x)
        x = tf.keras.layers.Dense(self.N_CLASS, activation='softmax')(x)  # N_CLASSE = 10

        super(InceptionResNetV2Fashion, self).__init__(inputs = inputs, outputs = x)

    def test_step(self, data):
        """The logic for one evaluation step.
        This method can be overridden to support custom evaluation logic.
        This method is called by `Model.make_test_function`.
        This function should contain the mathematical logic for one step of
        evaluation.
        This typically includes the forward pass, loss calculation, and metrics
        updates.
        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_test_function`, which can also be overridden.
        Arguments:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned.
        """
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        # BATCH (take care of OOM)
        y_pred = self(x, training=False)
        # Updates stateful loss metrics.
        self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        y_pred = np.squeeze(y_pred)
        y_ohe = np.squeeze(y)
        y_ohe = np.expand_dims(y_ohe, axis=1)
        y_ohe = enc.transform(y_ohe).toarray()
        r_sum = 0
        for c in range(self.N_CLASS):
            try:
                r = roc_auc_score(y_ohe[:, c], y_pred[:, c])
                #print('Label: ', y_labels[c], ', ', 'AUROC one vs rest = ', r)
                r_sum += r
            except ValueError:
                print('Label: ', y_labels[c], ', ', 'AUROC one vs rest = None!')

        mean_auroc = r_sum / self.N_CLASS
        logs_dict = {m.name: m.result() for m in self.metrics}
        logs_dict.update({'mean_auroc':mean_auroc})

        return logs_dict



class CMTensorBoardCallback(tf.keras.callbacks.TensorBoard):
  def __init__(self, file_writer, **kwargs):
    super(CMTensorBoardCallback, self).__init__(**kwargs)
    self.file_writer = file_writer
    self.epoch_step = 0

  def on_test_end(self, logs=None):
      # Use the model to predict the values from the validation dataset.
      test_pred_raw= self.model.predict(x_valid) #for large input
      # AUROC print
      y_pred = np.squeeze(test_pred_raw)
      y_ohe = np.squeeze(y_valid)
      y_ohe = np.expand_dims(y_ohe, axis=1)
      y_ohe = enc.transform(y_ohe).toarray()
      print('')
      r_array = []
      for c in range(N_CLASS):
          try:
              r = roc_auc_score(y_ohe[:, c], y_pred[:, c])
              print('Label: ', y_labels[c], ', ', 'AUROC one vs rest = ', r)
              r_array += r
          except ValueError:
              print('Label: ', y_labels[c], ', ', 'AUROC one vs rest = None!')

      test_pred = np.argmax(test_pred_raw, axis=1)
      # Calculate the confusion matrix.
      cm = confusion_matrix(y_valid, test_pred)
      # Log the confusion matrix as an image summary.
      figure = plot_confusion_matrix(cm, y_labels)
      cm_image = plot_to_image(figure)
      # Log the confusion matrix as an image summary.
      with self.file_writer.as_default():
          tf.summary.image("End epoch Confusion Matrix", cm_image, step=self.epoch_step, description=str(logs))
          
      self.epoch_step += 1

      self._pop_writer()

'''import sys
sys.setrecursionlimit(10000)'''

HP_REDUCE_SIZE = hp.HParam('reduce_size', hp.Discrete([1, 2, 3]))
#HP_DENSE_UNITS = hp.HParam('dense_units', hp.Discrete([1000, 500]))
#HP_DENSE_UNITS = hp.HParam('dense_units', hp.Discrete([100]))
HP_LR = hp.HParam('initial_lr', hp.Discrete([0.001, 0.01]))
#HP_LR = hp.HParam('initial_lr', hp.Discrete([0.001]))
#HP_INCEPTION_DROP_RATE = hp.HParam('inception_drop_rate', hp.Discrete([0.7, 0.5]))
#HP_DENSE_DROP_RATE = hp.HParam('dense_drop_rate', hp.Discrete([0.7, 0.5]))
HP_DROP_RATE = hp.HParam('drop_rate', hp.Discrete([0.8, 0.65]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
#reference metric must be equal to that displayed in tensorboard (tf.summary...)
METRIC_VALID_ACCURACY = hp.Metric("epoch_accuracy", group="validation", display_name="valid_accuracy")
METRIC_MEAN_AUROC = hp.Metric("epoch_mean_auroc", group="validation", display_name="valid_mean_auroc")
METRIC_TRAIN_ACCURACY = hp.Metric("epoch_accuracy", group="train", display_name="train_accuracy")


with tf.summary.create_file_writer(log_dir).as_default():
  hp.hparams_config(
    hparams=[HP_REDUCE_SIZE,
             HP_DROP_RATE,
             HP_LR,
             HP_OPTIMIZER],
    metrics=[METRIC_VALID_ACCURACY,
             METRIC_MEAN_AUROC,
             METRIC_TRAIN_ACCURACY]
  )

for REDUCE_SIZE in HP_REDUCE_SIZE.domain.values:
    for DROP_RATE in HP_DROP_RATE.domain.values:
        for LR in HP_LR.domain.values:
            for OPTIMIZER in HP_OPTIMIZER.domain.values:
                hparams = {HP_REDUCE_SIZE:REDUCE_SIZE,
                           HP_DROP_RATE:DROP_RATE,
                           HP_LR:LR,
                           HP_OPTIMIZER:OPTIMIZER}
                dir_name = ' ' + 'REDUCE_SIZE-' + str(REDUCE_SIZE) + ' ' + \
                           'DROP_RATE-' + str(DROP_RATE) + ' ' + \
                            'LR-' + str(LR) + ' ' + 'OPTIMIZER-' + str(OPTIMIZER)
                print('HP ---> ', dir_name)
                tb_callback_hparam = hp.KerasCallback(log_dir + dir_name, hparams)
                file_writer = tf.summary.create_file_writer(log_dir + dir_name)
                tb_cm_callback = CMTensorBoardCallback(file_writer,
                                                       log_dir=(log_dir + dir_name))
                checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath + dir_name,
                                                                        save_weights_only=True,
                                                                        monitor='accuracy',
                                                                        mode='max',
                                                                        save_best_only=True)
                early_callback = tf.keras.callbacks.EarlyStopping(
                    monitor='accuracy', min_delta=0, patience=3, verbose=1,
                    mode='max'
                )
                model = InceptionResNetV2Fashion((28,28,1), N_CLASS,
                                                 REDUCE_SIZE,
                                                 DROP_RATE)
                loss = tf.keras.losses.SparseCategoricalCrossentropy()
                #boundaries = [100, 1000]
                #values = [LR, LR/10., LR/100.]
                #lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                                                LR,
                                                                                decay_steps=400,
                                                                                decay_rate=0.90,
                                                                                staircase=True)
                if OPTIMIZER == 'adam':
                    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
                else:
                    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
                metric = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

                model.compile(optimizer=optimizer, loss=loss, metrics=[metric],
                              run_eagerly=True)
                model.fit(img_generator.flow(x_train, y_train, batch_size=32, shuffle=True),
                          validation_data=img_generator.flow(x_valid, y_valid, batch_size=256, shuffle=False),
                          epochs=12, callbacks=[tb_callback_hparam, tb_cm_callback, checkpoint_callback, early_callback],
                          verbose=2)
                            #batch_size>1! (computing AUROC)
                            #checkpoint_callback is useless if epoch=1

