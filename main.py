import os
import numpy as np
import tensorflow as tf, keras
from sklearn.model_selection import train_test_split
from model import EnvNet
from train import train_model
from data_preprocess import make_frames, make_frames_folder
from tqdm.auto import tqdm


# Training and K fold cross validation
def results(
    frame_length, overlapping_fraction, folds, model_config, batch_size, epochs
):
    models = {}
    folders = os.listdir("./dataset/mini")
    try:
        dataset = np.load("dataset_checkpoint.npy")
    except:
        dataset = make_frames_folder(folders, frame_length, overlapping_fraction)
        
    for i in tqdm(range(len(dataset))):
        np.random.shuffle(dataset)
        X = dataset[:, 0:frame_length]
        Y = dataset[:, frame_length]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        Y = keras.utils.np_utils.to_categorical(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.1, random_state=42
        )
        train_data = {"x_train": X_train, "y_train": Y_train}
        model = EnvNet(10, model_config)
        history = train_model(
            model,
            keras.losses.mean_squared_logarithmic_error,
            keras.optimizers.Adadelta(),
            train_data,
            epochs,
            batch_size,
        )
        test_results = model.evaluate(X_test, Y_test)
        models[i] = [model, history, test_results]
    return models


if __name__ == "__main__":
    tf.keras.backend.clear_session()
    tf.autograph.set_verbosity(5)
    frame_length, overlapping_fraction, folds, model_config, batch_size, epochs = (
        5,
        0.2,
        1,
        5,
        1,
        1,
    )
    results(frame_length, overlapping_fraction, folds, model_config, batch_size, epochs)
