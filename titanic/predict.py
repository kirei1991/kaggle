import tensorflow as tf
import titanic.data_module as dm


def get_compiled_model():
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(7, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    nn_model.compile(optimizer='adam',
                     loss='mean_squared_error',
                     metrics=['accuracy'])
    return nn_model


def predict_nn():
    dataset = tf.data.Dataset.from_tensor_slices((x_train.values, y_train.values))
    train_dataset = dataset.shuffle(len(x_train)).batch(1)

    model = get_compiled_model()
    model.fit(train_dataset, epochs=100)
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
    print("train_acc = ", train_acc)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print("test_acc = ", test_acc)

    y = model.predict(x_predict)
    dm.save_to_csv(y)


x_train, x_test, y_train, y_test, x_predict = dm.get_data()

if __name__ == '__main__':
    predict_nn()
