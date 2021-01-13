import tensorflow as tf

'''
# TWO OUTPUTS TEST
def get_model():
  input = tf.keras.layers.Input((6,1))
  dense = tf.keras.layers.Dense(3)(input)
  flat = tf.keras.layers.Flatten()(input)
  model = tf.keras.Model(inputs=input, outputs=dense)
  model_local = tf.keras.Model(inputs=input, outputs=flat)
  return model, model_local

model, model_identity = get_model()
x = tf.random.normal((1,6,1))
print('x: ', x)
print(model(x))
print(model_identity(x))
'''

'''
# TWO COMPLEX OUTPUTS TEST
class DoubleOutModel():
    def __init__(self):
        input = tf.keras.layers.Input((28, 28, 1))
        conv = tf.keras.layers.Conv2D(3, (3, 3), strides=1)(input)
        self.model_node = tf.keras.Model(inputs=input, outputs=conv)

        node_shape = (conv.shape[1], conv.shape[2], 1)
        input_local = tf.keras.Input(node_shape)
        flat_local = tf.keras.layers.Flatten()(input_local)
        dense_local = tf.keras.layers.Dense(5, activation='softmax')(flat_local)
        self.model_local = tf.keras.Model(inputs=input_local, outputs=dense_local)

        input_global = tf.keras.Input(conv.shape[1:])
        flat = tf.keras.layers.Flatten()(input_global)
        dense = tf.keras.layers.Dense(10, activation='softmax')(flat)
        self.model_global = tf.keras.Model(inputs=input_global, outputs=dense)

    def __call__(self, x):
        y_node = self.model_node(x)
        y_global = self.model_global(y_node)
        x_local = tf.expand_dims(y_node[:,:,:,0], axis=-1)
        y_local = self.model_local(x_local)
        return y_global, y_local

model = DoubleOutModel()
x = tf.random.normal((1,28,28,1))
y, y_local = model(x)
print('y: ',y)
print('y_local: ', y_local)'
'''

# TRAIN STEP TWO COMPLEX OUTPUTS
class DoubleOutModel(tf.keras.Model):
    def __init__(self):
        super(DoubleOutModel, self).__init__()

        input = tf.keras.layers.Input((28, 28, 1))
        conv = tf.keras.layers.Conv2D(3, (3, 3), strides=1)(input)
        self.model_node = tf.keras.Model(inputs=input, outputs=conv)

        node_shape = (conv.shape[1], conv.shape[2], 1)
        input_local = tf.keras.Input(node_shape)
        flat_local = tf.keras.layers.Flatten()(input_local)
        dense_local = tf.keras.layers.Dense(5, activation='linear')(flat_local)
        output_local = tf.keras.layers.Dense(1, activation='linear')(dense_local)
        self.model_local = tf.keras.Model(inputs=input_local, outputs=output_local)

        input_global = tf.keras.Input(conv.shape[1:])
        flat = tf.keras.layers.Flatten()(input_global)
        dense = tf.keras.layers.Dense(10, activation='relu')(flat)
        output_global = tf.keras.layers.Dense(1, activation='relu')(dense)
        self.model_global = tf.keras.Model(inputs=input_global, outputs=output_global)

        # Compile models (ONLY global and local model)
        self.model_local.compile(loss='mse')
        self.model_global.compile(loss='mse')

    def call(self, x, training=False): #training is necessary in case there are some BN/Dropout layers
        y_node = self.model_node(x, training)
        y_global = self.model_global(y_node, training)
        x_local = tf.expand_dims(y_node[:,:,:,0], axis=-1)
        y_local = self.model_local(x_local, training)
        return y_global, y_local

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_global_pred, y_local_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            loss_global = self.model_global.compiled_loss(y, y_global_pred)
            loss_local = self.model_local.compiled_loss(y, y_local_pred)
            loss = loss_global * 0.5 + loss_local * 0.5

        # Compute gradients
        trainable_vars = self.trainable_variables + self.model_global.trainable_variables + \
                         self.model_local.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return a dict mapping metric names to current value
        losses = {'loss_global':loss_global, 'loss_local':loss_local, 'loss':loss}
        return losses

    def test_step(self, data):
        pass


model = DoubleOutModel()
x_train = tf.random.normal((60,28,28,1))
y_train = tf.random.normal((60,1))

tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs/attention_example_weights_check', histogram_freq=1)
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer)
model.fit(x_train, y_train, batch_size=3, epochs=10, callbacks=[tb_callback])

