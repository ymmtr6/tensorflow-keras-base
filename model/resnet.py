# -*- coding:utf-8 -*-
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model
import tensorflow.python.keras

"""
ResnetBuilder

本来は100層程度存在するが，決め打ちで作成したので低層になっている
"""
class ResnetBuilder(object):

    """ build """
    def build(self, input_shape=(256,256, 3), class_num=3):
        self.model = self.__build_layer(input_shape, class_num)
        return self.model

    """ save """
    def save(self, filename):
        self.model.save(filename)

    """ load """
    def load(self, filename):
        self.model = keras.models.load_model(filepath)
        return self.model

    """ getter """
    def getModel(self):
        return self.model

    """
    residual_blockの実装
    """
    def __residual_block1(self, input_ts, filter, stride=1, kernel=(3,3)):
        """ ResidualBlock構成 """
        x = BatchNormalization()(input_ts)
        x = Activation('relu')(x)
        x = Conv2D(
        filter,
        kernel,
        stride,
        padding='same'
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(
        filter,
        kernel,
        stride,
        padding='same'
        )(x)
        return Add()([x, input_ts])

    """
    ResidualBlockの実装2．
    strides=(2,2)に指定したConv2DLayerで，出力の画像サイズを小さくする．
    """
    def __residual_block2(self, input_ts, filter, stride=1,kernel=(3,3)):
        """ ResidualBlock構成 """
        x = BatchNormalization()(input_ts)
        x = Activation('relu')(x)
        x = Conv2D(
        filter,
        kernel,
        strides=(2, 2),
        padding='same'
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(
        filter,
        kernel,
        stride,
        padding='same'
        )(x)
        """ residual shortcut """
        y = Conv2D(
        filter,
        kernel,
        strides=(2, 2),
        padding='same'
        )(input_ts)
        return Add()([x, y])

    """
    モデルの作成．
    """
    def __build_layer(self, input_shape=(256,256, 3), class_num=3):
        # Encoder
        input_ts = Input(shape=input_shape, name='input')
        # 入力を[0, 1]の範囲に正規化
        # x = Lambda(lambda a : a / 255.)(input_ts)
        # Conv2D
        x = Conv2D(
            64,
            (3, 3),
            strides=(2, 2),
            padding='same'
        )(input_ts)
        # BN, ReLU
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # MaxPool ( 分岐 )
        y = MaxPooling2D(
            (2, 2),
            padding='valid'
        )(x)
        # Conv2D
        x = Conv2D(
            64,
            (3, 3),
            strides=1,
            padding='same'
        )(y)
        # BN, ReLU
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # Conv2D
        x = Conv2D(
            64,
            (3, 3),
            strides=1,
            padding='same'
        )(x)
        # shortcut 合流
        x = Add()([x, y])
        # Residual Block ( input_shape, filter)
        x = self.__residual_block1(x, 64)
        x = self.__residual_block2(x, 128)
        x = self.__residual_block1(x, 128)
        x = self.__residual_block2(x, 256)
        x = self.__residual_block1(x, 256)
        x = self.__residual_block2(x, 512)
        x = self.__residual_block1(x, 512)
        # BN, ReLU
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # argPool filter(8, 8)
        x = AveragePooling2D(
            (8, 8),
            strides = None,
            padding='same',
            data_format=None
        )(x)
        # Flatten, Dense
        x = Flatten()(x)
        x = Dense(
            class_num,
            activation='softmax',
            name='output_dence'
        )(x)
        # Model generate
        model_gen = Model(
            inputs=[input_ts],
            outputs=[x]
        )
        return model_gen


if __name__ == '__main__':
    builder = ResnetBuilder()
    model = builder.build()
    model.summary()
