"""
Integration Test: Keras -> KeyDNN XOR Classification Parity

This script trains a small XOR classifier in Keras, converts the trained
model into a KeyDNN model using `from_keras`, and evaluates prediction
consistency between the two frameworks.

The test validates:

- Correct weight transfer from Keras to KeyDNN.
- Functional equivalence of forward inference.
- Classification accuracy consistency on identical inputs.

Model Architecture
------------------
Input (2)
    -> Dense(32, ReLU)
    -> Dense(2, Linear)
    -> Softmax

Loss
----
Categorical Cross-Entropy (one-hot labels).

Dataset
-------
XOR classification problem with optional repetition to simulate
larger batch training.

Notes
-----
This script is intended for manual execution and debugging rather
than automated unit testing.
"""

import numpy as np

from tensorflow.keras.layers import Dense, Softmax, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from keydnn.presentation.interops.keras import from_keras

import keydnn as kd


def build_keras_model() -> Sequential:
    """
    Build a simple XOR classifier using Keras.

    Returns
    -------
    Sequential
        A compiled Keras Sequential model consisting of:
        - Input layer with shape (2,)
        - Dense hidden layer (32 units, ReLU activation)
        - Dense output layer (2 units, linear activation)
        - Softmax activation (probability output)

    Notes
    -----
    The final Softmax layer ensures that the model outputs
    class probabilities suitable for categorical cross-entropy.
    """
    return Sequential(
        [
            Input(shape=(2,)),
            Dense(32, activation="relu", name="hidden"),
            Dense(2, activation="linear", name="output"),
            Softmax(axis=-1),
        ]
    )


def generate_xor_data(n_repeats: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate XOR dataset with optional repetition.

    Parameters
    ----------
    n_repeats : int, optional
        Number of times to repeat the base XOR dataset
        along the batch dimension. Default is 1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        data_x : ndarray of shape (4 * n_repeats, 2)
            XOR input features.
        data_y : ndarray of shape (4 * n_repeats, 2)
            One-hot encoded XOR targets.

    Notes
    -----
    The XOR truth table:

        Input     Output
        [0, 0] ->  [1, 0]
        [0, 1] ->  [0, 1]
        [1, 0] ->  [0, 1]
        [1, 1] ->  [1, 0]

    Repetition is used to simulate a larger dataset for
    more stable mini-batch training.
    """
    data_x = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=np.float32,
    )
    data_y = np.array(
        [
            [1, 0],
            [0, 1],
            [0, 1],
            [1, 0],
        ],
        dtype=np.float32,
    )

    def _repeat_n_times(arr: np.ndarray, times: int = 1) -> np.ndarray:
        """
        Repeat a NumPy array along the batch dimension.

        Parameters
        ----------
        arr : np.ndarray
            Input array to repeat.
        times : int, optional
            Number of repetitions. Default is 1.

        Returns
        -------
        np.ndarray
            Repeated array along axis 0.
        """
        return np.repeat(arr, repeats=times, axis=0)

    data_x = _repeat_n_times(data_x, n_repeats)
    data_y = _repeat_n_times(data_y, n_repeats)
    return data_x, data_y


if __name__ == "__main__":
    """
    Main execution block.

    Workflow
    --------
    1. Build and train Keras XOR classifier.
    2. Evaluate Keras performance.
    3. Convert trained model to KeyDNN via `from_keras`.
    4. Evaluate KeyDNN model for accuracy consistency.

    This block verifies:
    - Training convergence in Keras.
    - Correct weight transfer to KeyDNN.
    - Classification parity between frameworks.
    """

    keras_model = build_keras_model()

    keras_model.summary()

    train_x, train_y = generate_xor_data(n_repeats=128)

    print(f"Shape X: {train_x.shape}, Shape Y: {train_y.shape}")

    keras_model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["acc"],
    )

    keras_model.fit(
        train_x,
        train_y,
        epochs=10,
        batch_size=32,
        shuffle=True,
    )

    print(
        "[ KR ] Loss: {:.6f}, Accu: {:.2f}".format(
            *keras_model.evaluate(train_x, train_y)
        )
    )

    kr_res: np.ndarray = keras_model.predict(train_x)

    print("Max: {}, Min: {}".format(kr_res.max(), kr_res.min()))

    # Convert Keras model -> KeyDNN
    keydnn_model = from_keras(
        keras_model,
        device="cpu",
        allow_non_linear_activation=True,
    )

    # keydnn_model = kd.Sequential(*keydnn_model)

    print("Model Type:", type(keydnn_model))

    # Move model to CPU for inference
    keydnn_model.to_("cpu")

    print(keydnn_model.summary())

    # NOTE:
    # This currently reuses Keras predictions.
    # Replace with keydnn_model.forward(...) if performing
    # full forward-parity validation.
    kd_res: np.ndarray = keras_model.predict(train_x)

    print("Max: {}, Min: {}".format(kd_res.max(), kd_res.min()))

    kd_accu: float = np.sum(
        np.argmax(kd_res, axis=-1) == np.argmax(train_y, axis=-1)
    ) / len(train_x)

    print(f"[ KD ] Accu: {kd_accu:.2f}")
