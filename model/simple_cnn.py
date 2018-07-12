from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model
import tensorflow.python.keras
import tensorflow as tf

class SimpleCNNBuilder(object):

    def build(self, input_shape=(28, 28, 1), class_num=10):
        self.model = self.__build_layer(input_shape, class_num)
        return self.model

    def build_multi_gpu(self, input_shape=(28, 28, 1), class_num=10, device="/cpu:0"):
        with tf.device("/cpu:0"):
            self.model = self.__build_layer(input_shape, class_num)
            return self.model

    def save(self, file):
        self.model.save(file)

    def load(self, file):
        self.model = keras.models.load_model(file)

    def __build_layer(self, input_shape, class_num):
        input_ts = Input(shape=input_shape, name='input')

        x = Conv2D( 32, (3, 3), strides=1, padding='same')(input_ts)
        x = Conv2D( 64, (3, 3), strides=1, padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(class_num, activation='softmax')(x)

        # Model generate
        model_gen = Model(
            inputs=[input_ts],
            outputs=[x]
        )
        return model_gen
