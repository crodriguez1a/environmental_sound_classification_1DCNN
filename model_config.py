# Model architecture configuration file
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Dense,
    Flatten,
    BatchNormalization,
    Dropout,
    Activation,
)
from gammatone_init import GammatoneInit


class FeatureBlock1(Model):
    def __init__(self):
        super(FeatureBlock1, self).__init__()
        self.l1 = Conv1D(filters=16, kernel_size=64, strides=2)
        self.l2 = Activation(activation="relu")
        self.l3 = BatchNormalization()
        self.l4 = MaxPooling1D(pool_size=8, strides=8)
        self.l5 = Activation(activation="relu")
        self.l6 = Conv1D(filters=32, kernel_size=32, strides=2)
        self.l7 = Activation(activation="relu")
        self.l8 = BatchNormalization()
        self.l9 = MaxPooling1D(pool_size=8, strides=8)
        self.l10 = Activation(activation="relu")
        self.l11 = Conv1D(filters=64, kernel_size=16, strides=2)
        self.l12 = Activation(activation="relu")
        self.l13 = BatchNormalization()
        self.l14 = Conv1D(filters=128, kernel_size=8, strides=2)
        self.l15 = Activation(activation="relu")
        self.l16 = BatchNormalization()
        self.l17 = Conv1D(filters=256, kernel_size=4, strides=2)
        self.l18 = Activation(activation="relu")
        self.l19 = BatchNormalization()
        self.l20 = MaxPooling1D(pool_size=4, strides=4)
        self.l21 = Activation(activation="relu")

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        x = self.l11(x)
        x = self.l12(x)
        x = self.l13(x)
        x = self.l14(x)
        x = self.l15(x)
        x = self.l16(x)
        x = self.l17(x)
        x = self.l18(x)
        x = self.l19(x)
        x = self.l20(x)
        x = self.l21(x)
        return x


class FeatureBlock2(Model):
    def __init__(self):
        super(FeatureBlock2, self).__init__()
        self.l1 = Conv1D(filters=16, kernel_size=64, strides=2)
        self.l2 = Activation(activation="relu")
        self.l3 = BatchNormalization()
        self.l4 = MaxPooling1D(pool_size=8, strides=8)
        self.l5 = Activation(activation="relu")
        self.l6 = Conv1D(filters=32, kernel_size=32, strides=2)
        self.l7 = Activation(activation="relu")
        self.l8 = BatchNormalization()
        self.l9 = MaxPooling1D(pool_size=8, strides=8)
        self.l10 = Activation(activation="relu")
        self.l11 = Conv1D(filters=64, kernel_size=16, strides=2)
        self.l12 = Activation(activation="relu")
        self.l13 = BatchNormalization()
        self.l14 = Conv1D(filters=128, kernel_size=8, strides=2)
        self.l15 = Activation(activation="relu")
        self.l16 = BatchNormalization()
        self.l17 = Conv1D(filters=256, kernel_size=4, strides=2)
        self.l18 = Activation(activation="relu")
        self.l19 = BatchNormalization()
        self.l20 = MaxPooling1D(pool_size=4, strides=4)
        self.l21 = Activation(activation="relu")

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        x = self.l11(x)
        x = self.l12(x)
        x = self.l13(x)
        x = self.l14(x)
        x = self.l15(x)
        x = self.l16(x)
        x = self.l17(x)
        x = self.l18(x)
        x = self.l19(x)
        x = self.l20(x)
        x = self.l21(x)
        return x


class FeatureBlock3(Model):
    def __init__(self):
        super(FeatureBlock3, self).__init__()
        self.l1 = Conv1D(filters=16, kernel_size=64, strides=2)
        self.l2 = Activation(activation="relu")
        self.l3 = BatchNormalization()
        self.l4 = MaxPooling1D(pool_size=8, strides=8)
        self.l5 = Activation(activation="relu")
        self.l6 = Conv1D(filters=32, kernel_size=32, strides=2)
        self.l7 = Activation(activation="relu")
        self.l8 = BatchNormalization()
        self.l9 = MaxPooling1D(pool_size=8, strides=8)
        self.l10 = Activation(activation="relu")
        self.l11 = Conv1D(filters=64, kernel_size=16, strides=2)
        self.l12 = Activation(activation="relu")
        self.l13 = BatchNormalization()
        self.l14 = Conv1D(filters=128, kernel_size=8, strides=2)
        self.l15 = Activation(activation="relu")
        self.l16 = BatchNormalization()

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        x = self.l11(x)
        x = self.l12(x)
        x = self.l13(x)
        x = self.l14(x)
        x = self.l15(x)
        x = self.l16(x)
        return x


class FeatureBlock4(Model):
    def __init__(self):
        super(FeatureBlock4, self).__init__()
        self.l1 = Conv1D(
            filters=64,
            kernel_size=512,
            strides=1,
            kernel_initializer=GammatoneInit(16000, 100, 2),
        )
        self.l1.trainable = False
        self.l2 = Activation(activation="relu")
        self.l3 = BatchNormalization()
        self.l4 = MaxPooling1D(pool_size=8, strides=8)
        self.l5 = Activation(activation="relu")
        self.l6 = Conv1D(filters=32, kernel_size=32, strides=2)
        self.l7 = Activation(activation="relu")
        self.l8 = BatchNormalization()
        self.l9 = MaxPooling1D(pool_size=8, strides=8)
        self.l10 = Activation(activation="relu")
        self.l11 = Conv1D(filters=64, kernel_size=16, strides=2)
        self.l12 = Activation(activation="relu")
        self.l13 = BatchNormalization()
        self.l14 = Conv1D(filters=128, kernel_size=8, strides=2)
        self.l15 = Activation(activation="relu")
        self.l16 = BatchNormalization()

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        x = self.l11(x)
        x = self.l12(x)
        x = self.l13(x)
        x = self.l14(x)
        x = self.l15(x)
        x = self.l16(x)
        return x


class FeatureBlock5(Model):
    def __init__(self):
        super(FeatureBlock5, self).__init__()
        self.l1 = Conv1D(filters=16, kernel_size=64, strides=2)
        self.l2 = Activation(activation="relu")
        self.l3 = BatchNormalization()
        self.l4 = MaxPooling1D(pool_size=8, strides=8)
        self.l5 = Activation(activation="relu")
        self.l6 = Conv1D(filters=32, kernel_size=32, strides=2)
        self.l7 = Activation(activation="relu")
        self.l8 = BatchNormalization()
        self.l9 = MaxPooling1D(pool_size=8, strides=8)
        self.l10 = Activation(activation="relu")
        self.l11 = Conv1D(filters=64, kernel_size=16, strides=2)
        self.l12 = Activation(activation="relu")
        self.l13 = BatchNormalization()

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        x = self.l11(x)
        x = self.l12(x)
        x = self.l13(x)
        return x


class FeatureBlock6(Model):
    def __init__(self):
        super(FeatureBlock6, self).__init__()
        self.l1 = Conv1D(filters=16, kernel_size=32, strides=2)
        self.l2 = Activation(activation="relu")
        self.l3 = BatchNormalization()
        self.l4 = MaxPooling1D(pool_size=2, strides=2)
        self.l5 = Activation(activation="relu")
        self.l6 = Conv1D(filters=32, kernel_size=16, strides=2)
        self.l7 = Activation(activation="relu")
        self.l8 = BatchNormalization()
        self.l9 = MaxPooling1D(pool_size=2, strides=2)
        self.l10 = Activation(activation="relu")
        self.l11 = Conv1D(filters=64, kernel_size=8, strides=2)
        self.l12 = Activation(activation="relu")
        self.l13 = BatchNormalization()

    def call(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)
        x = self.l11(x)
        x = self.l12(x)
        x = self.l13(x)
        x = self.l14(x)
        x = self.l15(x)
        x = self.l16(x)
        return x
