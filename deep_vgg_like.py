from losslandscape import *
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2
from keras.layers import AveragePooling2D, Input, Flatten
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
import keras
import os
import numpy as np

from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import Adam, SGD
from keras.models import Sequential
from keras.utils import to_categorical
from keras.regularizers import l1, l2

from keras.datasets import cifar10

num_samples = 60000
num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


def vgg_layer(inputs,
              num_filters=16,
              kernel_size=3,
              strides=1,
              activation='relu',
              batch_normalization=True,
              conv_first=True):
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(0.0005))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def vgg_like(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    y = vgg_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = vgg_layer(inputs=y,
                          num_filters=num_filters,
                          strides=strides)
            y = vgg_layer(inputs=y,
                          num_filters=num_filters,
                          activation=None)
            # if stack > 0 and res_block == 0:  # first layer but not first stack
            #    # linear projection residual shortcut connection to match
            #    # changed dims
            y = Activation('relu')(y)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    y = AveragePooling2D(pool_size=8)(y)
    y = Flatten()(y)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def train_network(model, batch_size=128):
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    datagen.fit(x_train)

    def schedule(epoch, lr):
        epoch += 1  # epoch as arg is indexed from 0
        if epoch == 150 or epoch == 225 or epoch == 275:
            return lr / 10
        return lr

    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                  validation_data=(x_test, y_test),
                                  epochs=300,
                                  verbose=2,
                                  workers=4,
                                  steps_per_epoch=x_train.shape[0] /
                                  batch_size,
                                  callbacks=[LearningRateScheduler(schedule, verbose=0)])
    return history


if __name__ == '__main__':
    depth = 56
    input_shape = x_train.shape[1:]

    model = vgg_like(input_shape=input_shape, depth=depth)

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01, momentum=0.9,
                                decay=0.0, nesterov=True),
                  metrics=['categorical_accuracy'])

    model.summary()

    history = train_network(model)

    plot_loss_3D(model, ("levels", "3d"), x_test, y_test, number_of_points=47)
