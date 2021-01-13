import tensorflow as tf
import itertools
import io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

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

def image_grid(data, labels, y_labels, labels_pred_prob=None):
  '''
  Args:
    data: imgs batch:[batch, h, w, 1]
    labels: [batch, 1]
  Returns:
    figure: matplotlib figure
  '''
  figure = plt.figure(figsize=(10,10))
  if len(data.shape)==4:
    batch = data.shape[0]
  if len(data.shape)==3:
    data = np.expand_dims(data, axis=0) #single batch case
    batch = data.shape[0]
  else:
    raise TypeError('data.shape is not correct (3 or 4): ', data.shape)

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
def plot_confusion_matrix(cm, class_names):
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