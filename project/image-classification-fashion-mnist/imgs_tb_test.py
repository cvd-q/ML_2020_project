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
import itertools


DATA_BASE_FOLDER = './'
x_train = np.load(os.path.join(DATA_BASE_FOLDER, 'train.npy'))
x_valid = np.load(os.path.join(DATA_BASE_FOLDER, 'validation.npy'))
x_test = np.load(os.path.join(DATA_BASE_FOLDER, 'test.npy'))
y_train = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'train.csv'))['class'].values
y_valid = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'validation.csv'))['class'].values
#only 30 images and test dataset won't be used
x_train = x_train[0:30]
x_valid = x_valid[0:30]
y_train = y_train[0:30]
y_valid = y_valid[0:30]

y_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
x_train = x_train.reshape(x_train.shape[0], 28, 28) # reconstruct images
x_valid = x_valid.reshape(x_valid.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)
x_train = tf.expand_dims(x_train, axis=-1)
x_valid = tf.expand_dims(x_valid, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)
img_generator = ImageDataGenerator(preprocessing_function=lambda x: x/255.)

import io
logdir = "./logs/images_tb_test/"
file_writer = tf.summary.create_file_writer(logdir)

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def image_grid(data, labels, labels_pred_prob=None):
  '''
  Args:
    data: imgs batch:[batch, h, w, 1]
    labels: [batch, 1]
  Returns:
    figure: matplotlib figure
  '''
  figure = plt.figure(figsize=(10,10))
  batch = data.shape[0]
  r_c = round(batch**(1/2)) + 1
  if labels_pred_prob == None:
    for i in range(batch):
    # Start next subplot.
        plt.subplot(r_c, r_c, i + 1, title=y_labels[labels[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(data[i], cmap=plt.cm.binary)
  else:
      preds = np.argmax(labels_pred_prob, axis=1)
      labels = np.squeeze(labels) #NECESSARY! (ABOVE WE WERE IN EAGER MODE!)
      for i in range(batch):
        # Start next subplot.
        plt.subplot(r_c, r_c, i + 1)
        plt.title(y_labels[labels[i]], loc='left')
        plt.title('Pred: ' + y_labels[preds[i]], loc='right', color='red')
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(data[i], cmap=plt.cm.binary)
  return figure

from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, class_names = y_labels):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.

  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Compute the labels from the normalized confusion matrix.
  labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure

'''
figure = image_grid()
# Convert to image and log
with file_writer.as_default():
  tf.summary.image("Training data", plot_to_image(figure), step=0)
'''

class TensoBoardCallbackBatchImgs (tf.keras.callbacks.TensorBoard):
  def __init__(self, **kwargs):
    super(TensoBoardCallbackBatchImgs, self).__init__(**kwargs)
    self.epoch_step = 1

  def on_test_end(self, logs=None):
      x, y, y_pred = self.model.test_batch
      figure = image_grid(x, y, y_pred)
      # Convert to image and log
      with file_writer.as_default():
          tf.summary.image("Prediction epoch", plot_to_image(figure), step=self.epoch_step)

      # Use the model to predict the values from the validation dataset.
      test_pred_raw = self.model.predict(x_valid)
      test_pred = np.argmax(test_pred_raw, axis=1)

      # Calculate the confusion matrix.
      cm = confusion_matrix(y_valid, test_pred)
      # Log the confusion matrix as an image summary.
      figure = plot_confusion_matrix(cm)
      cm_image = plot_to_image(figure)

      # Log the confusion matrix as an image summary.
      with file_writer.as_default():
          tf.summary.image("Confusion Matrix", cm_image, step=self.epoch_step)

      self.epoch_step += 1

      self._pop_writer()

  '''def on_epoch_end(self, epoch, logs=None):
    """Runs metrics and histogram summaries at epoch end."""
    x,y = self.model.train_batch
    figure = image_grid(x, y)
    # Convert to image and log
    with file_writer.as_default():
        tf.summary.image("Training data", plot_to_image(figure), step=self.epoch_step)
    self.epoch_step += 1

    self._log_epoch_metrics(epoch, logs)

    if self.histogram_freq and epoch % self.histogram_freq == 0:
      self._log_weights(epoch)

    if self.embeddings_freq and epoch % self.embeddings_freq == 0:
      self._log_embeddings(epoch)'''


class MyModel(Model):
    def __init__(self, input, output):
        super(MyModel, self).__init__(inputs=input, outputs=output)

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
        from tensorflow.python.keras.engine import data_adapter
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        y_pred = self(x, training=False)
        # Updates stateful loss metrics.
        self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses)

        self.test_batch = (x, y, y_pred)

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    '''def train_step(self, data):
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

        self.train_batch = (x, y)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

'''
def get_model():
  input = tf.keras.layers.Input((28,28,1))
  conv = tf.keras.layers.Conv2D(3,(3,3),strides=1)(input)
  flat = tf.keras.layers.Flatten()(conv)
  dense = tf.keras.layers.Dense(10, activation='softmax')(flat)
  return MyModel(input=input, output=dense)

model = get_model()
tb_img_batch_callback = TensoBoardCallbackBatchImgs(log_dir=logdir)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='sparse-accuracy')],
              run_eagerly=True) #IT'S NECESSARY IN MOST OF CASES WHEN CUSTOMIZING CALLBACKS (TEST MODE MAYBE NO)


model.fit(img_generator.flow(x_train, y_train, batch_size=3, shuffle=True),
          validation_data=img_generator.flow(x_valid, y_valid, batch_size=3, shuffle=False), epochs=3,
          callbacks=[tb_img_batch_callback])

