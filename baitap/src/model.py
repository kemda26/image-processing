import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
# from keras.preprocessing import image

def create_model(optimizer, learning_rate, input_shape=(32, 32, 3)):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, input_shape=input_shape, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, 3, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(10, activation='softmax'))

    lr = learning_rate
    optimizers = {"SGD": keras.optimizers.SGD(lr=lr), "RMSprop": keras.optimizers.RMSprop(lr=lr),
                  "Adadelta": keras.optimizers.Adadelta(lr=lr), "Adam": keras.optimizers.Adam(lr=lr)}

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizers[optimizer],
                  metrics=['accuracy'])

    return model