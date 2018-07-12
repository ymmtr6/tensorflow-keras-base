from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model
import tensorflow.python.keras

class VGG16Builder(object):

    def build(self, input_shape=(224,224, 3)):
        self.model = self.__build_layer(input_shape)
        return self.model

    def save(self, file):
        self.model.save(file)

    def load(self, file):
        self.model = keras.models.load_model(file)

    def __build_layer(self, input_shape):
        input_ts = Input(shape=input_shape, name='input')

        x = Conv2D( 64, (3, 3), strides=1, padding='same')(input_ts)
        x = Conv2D( 64, (3, 3), strides=1, padding='same')(x)
        x = MaxPooling2D( (2, 2), padding='valid')(x)
        x = Conv2D(128, (3, 3), strides=1, padding='same')(x)
        x = Conv2D(128, (3, 3), strides=1, padding='same')(x)
        x = MaxPooling2D( (2, 2), padding='valid')(x)
        x = Conv2D(256, (3, 3), strides=1, padding='same')(x)
        x = Conv2D(256, (3, 3), strides=1, padding='same')(x)
        x = Conv2D(256, (3, 3), strides=1, padding='same')(x)
        x = MaxPooling2D( (2, 2), padding='valid')(x)
        x = Conv2D(512, (3, 3), strides=1, padding='same')(x)
        x = Conv2D(512, (3, 3), strides=1, padding='same')(x)
        x = Conv2D(512, (3, 3), strides=1, padding='same')(x)
        x = MaxPooling2D( (2, 2), padding='valid')(x)
        x = Flatten()(x)
        x = Dence(4096)(x)
        x = Dence(4096)(x)
        x = Dence(1000, activation='softmax', name='predictions')(x)

        # Model generate
        model_gen = Model(
            inputs=[input_ts],
            outputs=[x]
        )
        return model_gen

if __name__ == "__main__":
    model = VGG16Builder().build()
    model.summary(



__
