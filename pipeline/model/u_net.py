#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *


def u_net(input_shape, loadWeight=False, weigthPath = None):
    inputs = keras.Input(shape=input_shape)
    '''down_1'''
    conv_1_1 = Conv2D(32,3,activation='relu',padding='same')(inputs)
    conv_1_1 = Conv2D(32,3,activation='relu',padding='same')(conv_1_1)

    '''down_2'''
    maxpooling_1 = MaxPool2D()(conv_1_1)
    conv_2_1 = Conv2D(64,3,activation='relu',padding='same')(maxpooling_1)
    conv_2_1 = Conv2D(64,3,activation='relu',padding='same')(conv_2_1)
    
    '''down_3'''
    maxpooling_2 = MaxPool2D()(conv_2_1)
    conv_3_1 = Conv2D(128,3,activation='relu',padding='same')(maxpooling_2)
    conv_3_1 = Conv2D(128,3,activation='relu',padding='same')(conv_3_1)

    '''down_4'''

    maxpooling_3 = MaxPool2D()(conv_3_1)
    conv_4_1 = Conv2D(256,3,activation='relu',padding='same')(maxpooling_3)
    conv_4_1 = Conv2D(256,3,activation='relu',padding='same')(conv_4_1)

    '''down_5'''
    maxpooling_4 = MaxPool2D()(conv_4_1)
    conv_5_1 = Conv2D(512,3,activation='relu',padding='same')(maxpooling_4)
    conv_5_1 = Conv2D(512,3,activation='relu',padding='same')(conv_5_1)
    
    '''up_1'''

    up_1 = Conv2D(256,2,activation='relu',padding='same')(UpSampling2D()(conv_5_1))
    up_1 = Conv2D(256,3,activation='relu',padding='same')(tf.concat([conv_4_1,up_1],axis=-1))
    up_1 = Conv2D(256,3,activation='relu',padding='same')(up_1)

    '''up_2'''

    up_2 = Conv2D(128,2,activation='relu',padding='same')(UpSampling2D()(up_1))
    up_2 = Conv2D(128,3,activation='relu',padding='same')(tf.concat([conv_3_1,up_2],axis=-1))
    up_2 = Conv2D(128,3,activation='relu',padding='same')(up_2)
    
    '''up_3'''
    
    up_3 = Conv2D(64,2,activation='relu',padding='same')(UpSampling2D()(up_2))
    up_3 = Conv2D(64,3,activation='relu',padding='same')(tf.concat([conv_2_1,up_3],axis=-1))
    up_3 = Conv2D(64,3,activation='relu',padding='same')(up_3)
    
    '''up_4'''
    
    up_4 = Conv2D(32,2,activation='relu',padding='same')(UpSampling2D()(up_3))
    up_4 = Conv2D(32,3,activation='relu',padding='same')(tf.concat([conv_1_1,up_4],axis=-1))
    up_4 = Conv2D(32,3,activation='relu',padding='same')(up_4)

    '''output_layers'''
    
    outputs = Conv2D(1,1,activation='relu',padding='same')(up_4)
    
    model = keras.Model(inputs=inputs,outputs=outputs)

    if loadWeight == True:
        model.load_weights(weigthPath)
    return model

