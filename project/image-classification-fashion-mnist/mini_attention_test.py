import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
from skimage.measure import label
import matplotlib.patches as patches

# Attention model utilities
def Attention_gen_patchs(ori_image, node_tensor, test=False):
    # feature map -> feature mask (using feature map to crop on the original image) -> crop -> patchs
    feature_conv = node_tensor.numpy()
    size_upsample = (28, 28)
    bz, h, w, nc = feature_conv.shape
    #for visualization
    minh_array = np.empty(0)
    minw_array = np.empty(0)
    maxh_array = np.empty(0)
    maxw_array = np.empty(0)
    for i in range(0, bz):
        feature = feature_conv[i]
        cam = feature.reshape((h * w, nc))
        cam = cam.sum(axis=1)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

        heatmap_bin = binImage(cv2.resize(cam_img, size_upsample))
        heatmap_maxconn = selectMaxConnect(heatmap_bin)
        heatmap_mask = heatmap_bin * heatmap_maxconn

        ind = np.argwhere(heatmap_mask != 0)
        minh = min(ind[:, 0])
        minw = min(ind[:, 1])
        maxh = max(ind[:, 0])
        maxw = max(ind[:, 1])
        minh_array = np.append(minh_array, minh)
        minw_array = np.append(minw_array, minw)
        maxh_array = np.append(maxh_array, maxh)
        maxw_array = np.append(maxw_array, maxw)

        # to ori image
        image = ori_image[i].numpy().reshape(28, 28, 1)
        #??????????????????????????????????????????????????????????????????????????????????????
        #image = image[int(28 * 0.334):int(28 * 0.667), int(28 * 0.334):int(28 * 0.667), :]
        #image = cv2.resize(image, size_upsample)
        image_crop = image[minh:maxh, minw:maxw]  #image was normalized before
        image_crop = cv2.resize(image_crop, size_upsample) #preprocess(Image.fromarray(image_crop.astype('uint8')).convert('RGB'))

        #img_variable = torch.autograd.Variable(image_crop.reshape(3, 224, 224).unsqueeze(0).cuda())
        if i==0:
            patchs_tensor = tf.expand_dims(image_crop, 0) #shape=[1, h, w]
        else:
            image_crop = tf.expand_dims(image_crop, 0)
            patchs_tensor = tf.concat((patchs_tensor, image_crop), 0)

    # patchs_tensor.shape=[bz, h, w]
    if test:
        return (tf.expand_dims(patchs_tensor, axis=-1), (minh_array, maxh_array, minw_array, maxw_array)) #visualization
    return tf.expand_dims(patchs_tensor, axis=-1)


def binImage(heatmap):
    _, heatmap_bin = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # t in the paper
    # _, heatmap_bin = cv2.threshold(heatmap , 178 , 255 , cv2.THRESH_BINARY)
    return heatmap_bin


def selectMaxConnect(heatmap):
    labeled_img, num = label(heatmap, connectivity=2, background=0, return_num=True)
    max_label = 0
    max_num = 0
    for i in range(1, num + 1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    if max_num == 0:
        lcc = (labeled_img == -1)
    lcc = lcc + 0
    return lcc


# PREPARE DATA
DATA_BASE_FOLDER = './'
x_train = np.load(os.path.join(DATA_BASE_FOLDER, 'train.npy'))
x_valid = np.load(os.path.join(DATA_BASE_FOLDER, 'validation.npy'))
x_test = np.load(os.path.join(DATA_BASE_FOLDER, 'test.npy'))
y_train = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'train.csv'))['class'].values
y_valid = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'validation.csv'))['class'].values
#only 500 images and test dataset won't be used
x_train = x_train[0:500]
x_valid = x_valid[0:500]
y_train = y_train[0:500]
y_valid = y_valid[0:500]

y_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
x_train = x_train.reshape(x_train.shape[0], 28, 28) # reconstruct images
x_valid = x_valid.reshape(x_valid.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)
x_train = tf.expand_dims(x_train, axis=-1)
x_valid = tf.expand_dims(x_valid, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)
img_generator = ImageDataGenerator(preprocessing_function=lambda x: x/255.)

# BUILD MINI BLOCK FOR TESTING
def get_block(input_shape = (28,28,1)):
    input = tf.keras.Input(input_shape)
    conv_1 = tf.keras.layers.Conv2D(3, (3,3), 2)(input)
    drop_1 = tf.keras.layers.Dropout(0.8)(conv_1)
    conv_2 = tf.keras.layers.Conv2D(3, (3,3))(drop_1)
    drop_2 = tf.keras.layers.Dropout(0.8)(conv_2)
    node = drop_2
    flat = tf.keras.layers.Flatten()(drop_2)

    return input, node, flat

def get_end_block(input_shape):
    input = tf.keras.Input(input_shape)
    dense_1 = tf.keras.layers.Dense(20, activation='relu')(input)
    drop = tf.keras.layers.Dropout(0.8)(dense_1)
    dense_2 = tf.keras.layers.Dense(10, activation='softmax')(drop)

    return input, dense_2

def get_fusion_model(input_shape):
    input = tf.keras.Input(input_shape)
    dense_1 = tf.keras.layers.Dense(32, activation='relu')(input)
    drop = tf.keras.layers.Dropout(0.8)(dense_1)
    dense_2 = tf.keras.layers.Dense(10, activation='softmax')(drop)

    return input, dense_2

class AttentionBasedModel (tf.keras.Model):
    def __init__(self, node_model, global_end_model, local_head_model, local_end_model, fusion_end_model):
        super(AttentionBasedModel, self).__init__()

        self.node_model = node_model
        self.global_end_model = global_end_model
        self.local_head_model = local_head_model
        self.local_end_model = local_end_model
        self.fusion_end_model = fusion_end_model

        self.global_end_model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                                        run_eagerly = True)
        self.local_end_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                        run_eagerly=True)
        self.fusion_end_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                        run_eagerly=True)
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy()

    def call(self, x, training=False):
        node_out_tensor = self.node_model(x, training)

        x_global = tf.keras.layers.Flatten()(node_out_tensor)
        y_global = self.global_end_model(x_global, training)

        if training:
            input_local = Attention_gen_patchs(x, node_out_tensor)
        else:
            input_local, points = Attention_gen_patchs(x, node_out_tensor, test=True)
        x_local = self.local_head_model(input_local, training)
        y_local = self.local_end_model(x_local, training)

        x_fusion = tf.keras.layers.Concatenate()([x_global, x_local])
        y_fusion = self.fusion_end_model(x_fusion, training)

        if training==False:
            return y_global, y_local, y_fusion, input_local, points

        return y_global, y_local, y_fusion, input_local


    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_global_pred, y_local_pred, y_fusion_pred, _ = self(x, training=True)  # Forward pass
            # Compute the loss value
            loss_global = self.global_end_model.compiled_loss(y, y_global_pred)
            loss_local = self.local_end_model.compiled_loss(y, y_local_pred)
            loss_fusion = self.fusion_end_model.compiled_loss(y, y_fusion_pred)
            loss = loss_global * 0.8 + loss_local * 0.1 + loss_fusion * 0.1

        # Compute gradients
        trainable_vars = self.node_model.trainable_variables +\
                         self.global_end_model.trainable_variables + \
                         self.local_head_model.trainable_variables + \
                         self.local_end_model.trainable_variables + \
                         fusion_end_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metric
        self.metric.reset_states()
        self.metric.update_state(y, y_global_pred)
        accuracy_global = self.metric.result()
        self.metric.reset_states()
        self.metric.update_state(y, y_local_pred)
        accuracy_local = self.metric.result()
        self.metric.reset_states()
        self.metric.update_state(y, y_fusion_pred)
        accuracy_fusion = self.metric.result()
        # Return a dict mapping metric names to current value
        dict = {'loss_global': loss_global,
                  'loss_local': loss_local,
                  'loss_fusion': loss_fusion,
                  'loss': loss,
                  'accuracy_global':accuracy_global,
                  'accuracy_local':accuracy_local,
                  'accuracy_fusion':accuracy_fusion}
        return dict

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

        y_global_pred, y_local_pred, y_fusion_pred, input_local, positions = self(x, training=False) #node_tensor 'has one channel'

        # Update metric
        self.metric.reset_states()
        self.metric.update_state(y, y_global_pred)
        accuracy_global = self.metric.result()
        self.metric.reset_states()
        self.metric.update_state(y, y_local_pred)
        accuracy_local = self.metric.result()
        self.metric.reset_states()
        self.metric.update_state(y, y_fusion_pred)
        accuracy_fusion = self.metric.result()

        self.test_batch = (x,
                           y,
                           y_global_pred,
                           y_local_pred,
                           y_fusion_pred,
                           input_local,
                           positions) #useful for tensorboard callback

        dict = {'accuracy_global': accuracy_global,
                'accuracy_local': accuracy_local,
                'accuracy_fusion': accuracy_fusion}
        return dict


input_model, node, fusion_global_tensor = get_block()
#node_shape = (node.shape[1], node.shape[2], 1) #ONLY 1 channel: TESTING. IT HAS TO BE DEFINED
local_model_input, _, fusion_local_tensor = get_block() #get_block(node_shape)

global_input, global_output = get_end_block(fusion_global_tensor.shape[1])
local_input, local_output = get_end_block(fusion_local_tensor.shape[1])
fusion_input, fusion_output = get_fusion_model(fusion_local_tensor.shape[1] + fusion_global_tensor.shape[1])

#create models
node_model = tf.keras.Model(inputs=input_model, outputs=node)
global_end_model = tf.keras.Model(inputs=global_input, outputs=global_output)
fusion_end_model = tf.keras.Model(inputs=fusion_input, outputs=fusion_output)
local_head_model = tf.keras.Model(inputs=local_model_input, outputs=fusion_local_tensor)
local_end_model = tf.keras.Model(inputs=local_input, outputs=local_output)

attention_model = AttentionBasedModel(node_model, global_end_model,
                                      local_head_model, local_end_model,
                                      fusion_end_model)

attention_model.compile(optimizer='adam', run_eagerly=True)
'''
x_train = tf.random.normal((60,28,28,1))
y_train = tf.constant(np.random.randint(low=0, high=9, size=(60,1)))
tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs/mini_attention_weights_check', histogram_freq=1)
attention_model.fit(x_train, y_train, batch_size=3, epochs=3, callbacks=[tb_callback])
'''
# TENSORBOAD CALLBACK
from tb_imgs_utilities import image_grid, plot_to_image, confusion_matrix, plot_confusion_matrix

logdir = "./logs/mini_attention_test/"
file_writer = tf.summary.create_file_writer(logdir)
class TensoBoardCallbackBatchImgs (tf.keras.callbacks.TensorBoard):
  def __init__(self, **kwargs):
    super(TensoBoardCallbackBatchImgs, self).__init__(**kwargs)
    self.epoch_step = 1

  def on_test_end(self, logs=None):
      # return to tensorboard ONLY last batch test data
      x, y, _, _, y_fusion_pred, input_local_imgs, positions= self.model.test_batch #we consider only fusion branch. node_img shape=[batch,h,w,1]
      figure = image_grid(x, y, y_labels=y_labels, labels_pred_prob=y_fusion_pred)
      # Convert to image and log
      with file_writer.as_default():
          tf.summary.image("Fusion end epoch prediction", plot_to_image(figure), step=self.epoch_step)

      # Use the model to predict the values from the validation dataset.
      _, _, test_pred_raw, _, _ = self.model.call(x_valid)
      test_pred = np.argmax(test_pred_raw, axis=1)

      # Calculate the confusion matrix.
      cm = confusion_matrix(y_valid, test_pred)
      # Log the confusion matrix as an image summary.
      figure = plot_confusion_matrix(cm, y_labels)
      cm_image = plot_to_image(figure)

      # Log the confusion matrix as an image summary.
      with file_writer.as_default():
          tf.summary.image("Fusion end epoch Confusion Matrix", cm_image, step=self.epoch_step)

      # Plot the first node image
      y = np.squeeze(y) # ACTIVATE EAGER MODE
      figure = plt.figure()
      ax_ori = figure.add_subplot(1, 2, 1)
      ax_ori.imshow(x[0], cmap=plt.cm.binary)
      minh, maxh, minw, maxw = positions
      rect = patches.Rectangle(xy=(minw[0], minh[0]),
                               width=maxw[0] - minw[0],
                               height=maxh[0]-minh[0],
                               linewidth=1, edgecolor='r', facecolor='none')
      ax_ori.add_patch(rect)
      ax_ori.set_title(y_labels[y[0]])
      ax_ori.set_xticks([])
      ax_ori.set_yticks([])
      ax_ori.grid(False)

      ax_crop = figure.add_subplot(1, 2, 2)
      ax_crop.imshow(input_local_imgs[0], cmap=plt.cm.binary)
      ax_crop.set_title(y_labels[y[0]] + '(cropped)')
      ax_crop.set_xticks([])
      ax_crop.set_yticks([])
      ax_crop.grid(False)
      with file_writer.as_default():
          tf.summary.image("input to local branch image(first one of the batch)",
                           plot_to_image(figure), step=self.epoch_step)

      self.epoch_step += 1

      self._pop_writer()

tb_img_batch_callback = TensoBoardCallbackBatchImgs(log_dir=logdir)
attention_model.fit(img_generator.flow(x_train, y_train, batch_size=12, shuffle=True),
                    validation_data=img_generator.flow(x_valid, y_valid, batch_size=12, shuffle=False),
                    epochs=10, callbacks=[tb_img_batch_callback]) #BATCH SIZE > 1 (IMAGE_GRID)