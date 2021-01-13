from ML_project_2020 import *

class TensoBoardCallbackEpoch (tf.keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super(TensoBoardCallbackEpoch, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        """Runs metrics and histogram summaries at epoch end."""
        logs['test_customization'] = tf.random.normal((1,)) # FOR TESTING
        self._log_epoch_metrics(epoch, logs)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_weights(epoch)

        if self.embeddings_freq and epoch % self.embeddings_freq == 0:
            self._log_embeddings(epoch)

from tensorboard.plugins.hparams import api as hp
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([10, 13, 16]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
METRIC_LOSS = hp.Metric('epoch_loss', group='train')
log_dir='./logs/simple_test/hparam_tuning'
with tf.summary.create_file_writer(log_dir).as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_OPTIMIZER],
    metrics=[METRIC_LOSS]
  )
x = tf.random.normal((10,3))
y = tf.random.normal((10,1))

i=1
for optimizer in HP_OPTIMIZER.domain.values:
    for num_units in HP_NUM_UNITS.domain.values:
        hparams = {HP_NUM_UNITS:num_units, HP_OPTIMIZER:optimizer}
        tb_callback = TensoBoardCallbackEpoch(log_dir=log_dir+str(i))
        tb_callback_hparam = hp.KerasCallback(log_dir+str(i), hparams)
        input = tf.keras.layers.Input(3)
        dense = tf.keras.layers.Dense(hparams[HP_NUM_UNITS])(input)
        dense_final = tf.keras.layers.Dense(1)(dense)
        model = tf.keras.Model(inputs=input, outputs=dense_final)
        model.compile(optimizer=hparams[HP_OPTIMIZER], loss='mse')
        history = model.fit(x, y, epochs=8, callbacks=[tb_callback, tb_callback_hparam])
        i+=1