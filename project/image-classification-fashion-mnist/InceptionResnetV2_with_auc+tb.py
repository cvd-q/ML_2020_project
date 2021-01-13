from ML_project_2020 import *

N_CLASS = 10
enc = OneHotEncoder(handle_unknown='ignore')
cl = np.arange(N_CLASS).reshape((N_CLASS,1))
enc.fit(cl)

DATA_BASE_FOLDER = './'
x_train = np.load(os.path.join(DATA_BASE_FOLDER, 'train.npy'))
x_valid = np.load(os.path.join(DATA_BASE_FOLDER, 'validation.npy'))
x_test = np.load(os.path.join(DATA_BASE_FOLDER, 'test.npy'))
y_train = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'train.csv'))['class'].values
y_valid = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'validation.csv'))['class'].values
y_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
x_train = x_train.reshape(x_train.shape[0], 28, 28) # reconstruct images
x_valid = x_valid.reshape(x_valid.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)
x_train = tf.expand_dims(x_train, axis=-1)
x_valid = tf.expand_dims(x_valid, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)
img_generator = ImageDataGenerator(preprocessing_function=lambda x: x/255.)

class MultiClassAucCallback (tf.keras.callbacks.Callback):
    def on_test_end(self, logs=None):
        # ALREADY IN EAGER MODE!!!
        y_pred = self.model.predict(img_generator.flow(x_valid, y_valid, batch_size=32, shuffle=False))
        y_pred = np.squeeze(y_pred)
        y_ohe = y_valid.squeeze()
        y_ohe = np.expand_dims(y_ohe, axis=1)
        y_ohe = enc.transform(y_ohe).toarray()
        print('')
        for c in range(N_CLASS):
            try:
                r = roc_auc_score(y_ohe[:, c], y_pred[:, c])
                print('Label: ', y_labels[c], ', ', 'AUROC one vs rest = ', r)
            except ValueError:
                print('Label: ', y_labels[c],', ', 'AUROC one vs rest = None!')

auc_callback = MultiClassAucCallback()
tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs/simple_test')

fashion_model = InceptionResNetV2Fashion((28,28,1), N_CLASS)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
fashion_model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='sparse-accuracy')])
fashion_model.fit(img_generator.flow(x_train[0:3], y_train[0:3], batch_size=32, shuffle=True),
                    validation_data=img_generator.flow(x_valid[0:3], y_valid[0:3], batch_size=32, shuffle=False), epochs=8, callbacks=[auc_callback, tb_callback])
