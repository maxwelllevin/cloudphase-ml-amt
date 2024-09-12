# Trains the CNN model with the best settings we found. To

import os
import warnings
from pathlib import Path

import numpy as np  # type: ignore

# Do this above the tensorflow imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hide info and warnings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # should fix most memory issues


import keras
import tensorflow as tf
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    LeakyReLU,
    SpatialDropout2D,
    concatenate,
)
from keras.models import Model
from keras.optimizers import Adam

warnings.simplefilter("ignore")


X_train_path = "../preprocessing/data/cnn_inputs/X_train.npy"
y_train_path = "../preprocessing/data/cnn_inputs/y_train.npy"
X_valid_path = "../preprocessing/data/cnn_inputs/X_valid.npy"
y_valid_path = "../preprocessing/data/cnn_inputs/y_valid.npy"


def setup_gpu(index: int | slice | None = None):
    """Restrict TensorFlow to only use one GPU. Use a slice to use several. BE KIND."""
    if index is None:
        index = slice(None, None)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[index], "GPU")
            if isinstance(index, slice):
                for gpu in gpus[index]:
                    tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
            print(f"{index=}, {gpus[index]=}")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    return


def load_training_data() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Loads X_train, y_train, X_valid, and y_valid data for training."""
    # load onto CPU to avoid out-of-memory issues
    with tf.device("CPU"):  # type: ignore
        X_train = tf.convert_to_tensor(np.load(X_train_path))
        y_train = tf.convert_to_tensor(np.load(y_train_path))
        X_valid = tf.convert_to_tensor(np.load(X_valid_path))
        y_valid = tf.convert_to_tensor(np.load(y_valid_path))
    return X_train, y_train, X_valid, y_valid


def build_20240429_model(X) -> Model:
    """Builds the model named '20240429.213223' in the paper. This model employs a
    spatial 2D dropout layer at the start of the model to mimic real-world instrument
    failures/unavailability."""
    INPUT_SIZE = (None, None, X.shape[-1])  # X.shape[-1] is #features
    OUTPUT_CHANNELS = 8  # the number of output classes
    N_FILTERS = [64, 64, 64, 128, 128, 256]

    def unet_block(x, filters):
        x = Conv2D(filters, (3, 3), padding="same")(x)
        x = Dropout(rate=0.1)(x)
        x = LeakyReLU()(x)
        # -------------------------
        x = Conv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    x_in = Input(INPUT_SIZE)
    x = SpatialDropout2D(rate=0.125)(x_in)  # 1 instrument/channel missing
    d_out = [unet_block(x, N_FILTERS[0])]
    for i in range(1, len(N_FILTERS)):  # downsampling path
        x = Conv2D(
            N_FILTERS[i],
            (4, 4),
            strides=(2, 2),
            padding="same",
            activation="linear",
        )(d_out[-1])
        x = unet_block(x, N_FILTERS[i])
        d_out.append(x)
    x = d_out.pop(-1)
    for i in range(len(N_FILTERS) - 1, 0, -1):  # upsampling path
        x = Conv2DTranspose(
            N_FILTERS[i - 1],
            (4, 4),
            strides=(2, 2),
            padding="same",
            activation="linear",
        )(x)
        x = concatenate([x, d_out.pop(-1)])
        x = unet_block(x, N_FILTERS[i - 1])
    x_out = Conv2D(OUTPUT_CHANNELS, (1, 1), activation="softmax")(x)
    model = Model(inputs=x_in, outputs=x_out)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalCrossentropy(
                ignore_class=0,
                name="cloudy_crossentropy",
            ),
            keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.MeanIoU(
                num_classes=8,
                ignore_class=0,  # ignore clear sky for IOU metrics
                sparse_y_pred=False,
                sparse_y_true=True,
            ),
        ],
    )
    return model


def build_20240501_model(X) -> Model:
    """Builds the model named '20240501.090456' in the paper. This model does not use a
    spatial 2D dropout layer, but is otherwise the same as the 20240429 model."""
    INPUT_SIZE = (None, None, X.shape[-1])  # X.shape[-1] is #features
    OUTPUT_CHANNELS = 8  # the number of output classes
    N_FILTERS = [64, 64, 64, 128, 128, 256]

    def unet_block(x, filters):
        x = Conv2D(filters, (3, 3), padding="same")(x)
        x = Dropout(rate=0.1)(x)
        x = LeakyReLU()(x)
        # -------------------------
        x = Conv2D(filters, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    x_in = Input(INPUT_SIZE)
    x = x_in
    d_out = [unet_block(x, N_FILTERS[0])]
    for i in range(1, len(N_FILTERS)):  # downsampling path
        x = Conv2D(
            N_FILTERS[i],
            (4, 4),
            strides=(2, 2),
            padding="same",
            activation="linear",
        )(d_out[-1])
        x = unet_block(x, N_FILTERS[i])
        d_out.append(x)
    x = d_out.pop(-1)
    for i in range(len(N_FILTERS) - 1, 0, -1):  # upsampling path
        x = Conv2DTranspose(
            N_FILTERS[i - 1],
            (4, 4),
            strides=(2, 2),
            padding="same",
            activation="linear",
        )(x)
        x = concatenate([x, d_out.pop(-1)])
        x = unet_block(x, N_FILTERS[i - 1])
    x_out = Conv2D(OUTPUT_CHANNELS, (1, 1), activation="softmax")(x)
    model = Model(inputs=x_in, outputs=x_out)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalCrossentropy(
                ignore_class=0,
                name="cloudy_crossentropy",
            ),
            keras.metrics.SparseCategoricalAccuracy(),
            keras.metrics.MeanIoU(
                num_classes=8,
                ignore_class=0,  # ignore clear sky for IOU metrics
                sparse_y_pred=False,
                sparse_y_true=True,
            ),
        ],
    )
    return model


def train_model(
    model: Model,
    label: str,
    train: tuple[tf.Tensor, tf.Tensor],
    valid: tuple[tf.Tensor, tf.Tensor],
    optimize_strategy: tuple[str, str],
    epochs: int,
):
    X_train, y_train = train
    X_valid, y_valid = valid
    monitor, mode = optimize_strategy

    logs_dir = Path(f"./logs/{label}/")
    logs_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        X_train,
        y_train,
        batch_size=16,
        epochs=epochs,
        verbose=2,  # one line per epoch
        validation_data=(X_valid, y_valid),
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                mode=mode,
                patience=3,
                factor=0.2,
                min_lr=1e-8,  # type: ignore
                verbose=0,
            ),
            keras.callbacks.ModelCheckpoint(
                f"models/cnn.{label}.h5",
                monitor=monitor,
                mode=mode,
                verbose=0,
                save_best_only=True,
            ),
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                mode=mode,
                verbose=0,
                patience=10,
                restore_best_weights=True,
                start_from_epoch=10,
            ),
            keras.callbacks.TensorBoard(
                log_dir=logs_dir.as_posix(),
                # histogram_freq=1,
                write_graph=True,
                write_images=True,
                write_steps_per_second=False,
            ),
        ],
    )


def main():
    print("starting cnn training script...")

    print("setting up the gpu(s)...")
    setup_gpu()

    print("loading training and validation data...")
    X_train, y_train, X_valid, y_valid = load_training_data()

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        print("building the model...")
        model, label = build_20240501_model(X_train), "20240501"  # best model
        # model, label = build_20240429_model(X_train), "20240429"  # second-best model

        print("training the model...")
        train_model(
            model=model,
            train=(X_train, y_train),
            valid=(X_valid, y_valid),
            label=label,
            optimize_strategy=("val_mean_io_u", "max"),
            epochs=100,
        )


if __name__ == "__main__":
    main()
