from tensorflow import keras
from tensorflow.keras.layers import *


def load_model(input_size, load_weight=False, weight_path=None):
    """
    input_size: we use 128. 256 or 64 is optional.
    if load_weight is True,please setting weight_path as well.
    """
    def siamese_cnn_model(input_size):
        base = keras.Sequential([
            Conv2D(6, 3, input_shape=input_size,kernel_regularizer=keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Activation('relu'),   # cov_layer + BN + relu
            Conv2D(6, 3, kernel_regularizer=keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Activation('relu'),
            MaxPool2D(),
            Conv2D(12, 3, kernel_regularizer=keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(12, 3, kernel_regularizer=keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Activation('relu'),
            MaxPool2D(),
        ])

        left = keras.Input(input_size)

        right = keras.Input(input_size)

        out_1 = base(left)
        out_2 = base(right)

        layer = Lambda(lambda x: x[1]-x[0])
        concat = layer([out_1, out_2])
        con_v_1 = Conv2D(8, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(concat)
        pooling_1 = MaxPool2D((2, 2))(con_v_1)
        con_v_2 = Conv2D(8, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(pooling_1)
        pooling_2 = MaxPool2D((2, 2))(con_v_2)

        fl = Flatten()(pooling_2)

        fc = Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(fl)
        dropout = Dropout(0.2)(fc)
        fc = Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(dropout)
        fc = Dense(2, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.01))(fc)

        model = keras.Model(inputs=[left, right], outputs=fc)

        return model
    Model = siamese_cnn_model((input_size, input_size, 1))

    opt = keras.optimizers.Adam(lr=0.001)
    Model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if load_weight:
        Model.load_weights(weight_path)
    return Model

