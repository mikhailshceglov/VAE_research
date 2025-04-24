import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical


def load_synthetic(path_images='synthetic_images.pt', path_labels='synthetic_labels.pt'):
    images = torch.load(path_images)
    labels = torch.load(path_labels)
    images = images.numpy().astype('float32')
    images = np.transpose(images, (0, 2, 3, 1))
    labels = labels.numpy().astype('int')
    return images, labels


def load_cifar10_val():
    (_, _), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()
    x_val = x_val.astype('float32') / 255.0
    y_val = to_categorical(y_val, 10)
    return x_val, y_val


def build_model(input_shape=(32,32,3), num_classes=10):
    model = Sequential([
        # Первый блок свёртки
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Второй блок свёртки
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Третий блок свёртки
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Четвёртый блок свёртки
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Полносвязный слой
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=7.5e-4)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():

    x_train, y_train_raw = load_synthetic()
    y_train = to_categorical(y_train_raw, 10)
    x_val, y_val = load_cifar10_val()

    model = build_model(input_shape=x_train.shape[1:], num_classes=10)
    model.summary()

    history = model.fit(
        x_train, y_train,
        epochs=15,
        batch_size=32,
        validation_data=(x_val, y_val),
        verbose=1
    )

    model.save('classifier_on_synthetic.h5')
    np.save('history.npy', history.history)

if __name__ == '__main__':
    main()

