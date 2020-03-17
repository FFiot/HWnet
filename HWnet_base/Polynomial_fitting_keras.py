from util import *

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

POLYNOMIAL_NUM = 8

if __name__ == "__main__":
    def test_fucntion(x):
        return np.sin(x**2 * np.pi * 8)/2 - (x)**2 + 0.5

    x = (np.random.random(4096) - 0.5)*2
    target = test_fucntion(x).reshape((-1,1))
    data_train = np.concatenate([(x**(i+1)).reshape(-1, 1) for i in range(POLYNOMIAL_NUM)], axis=-1)
    plt_scatter(x, y_true=target)

    model = keras.Sequential([Dense(POLYNOMIAL_NUM, use_bias=False, activation='tanh'),
                              Dense(1, activation='tanh')])
    
    model.build(input_shape=(None, POLYNOMIAL_NUM))
    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.Huber())

    class plt_callback(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            predict = self.model(data_train)
            loss_mean = self.model.loss(predict, target).numpy().mean()
            title = 'loss:%0.6f'%(loss_mean)
            plt_scatter(x, predict, target, title)

        def on_epoch_end(self, epoch, logs={}):
            if (epoch+1)%1000 == 0:
                predict = self.model(data_train)
                title = 'epotch:%04d,loss:%0.6f'%(epoch+1, logs['loss'])
                plt_scatter(x, predict, target, title)

    model.fit(data_train, target, batch_size=128, epochs=10000, callbacks=[plt_callback()])
