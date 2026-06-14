"""Trainer module for Credit Risk Prediction TFX Pipeline.

This module contains the run_fn function used by the TFX Trainer component
to build, train, and export a DNN classifier for credit risk prediction.
"""

import tensorflow as tf
import tensorflow_transform as tft
from tfx_bsl.public import tfxio
from tensorflow.keras import layers  # pylint: disable=import-error,no-name-in-module

from modules.transform_module import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)

# Training parameters
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
TRAIN_STEPS = 500
EVAL_STEPS = 100
HIDDEN_UNITS = [128, 64, 32]
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.5


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies
    the model for serving.

    Args:
        model: The trained Keras model.
        tf_transform_output: The TFTransformOutput object.

    Returns:
        A serving function.
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns predictions from the model for serving."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )

        transformed_features = model.tft_layer(parsed_features)
        outputs = model(transformed_features)

        return {'outputs': outputs}

    return serve_tf_examples_fn


def _get_transform_feature_spec(tf_transform_output):
    """Get the feature spec for transformed features.

    Args:
        tf_transform_output: The TFTransformOutput object.

    Returns:
        A dict of transformed feature specs.
    """
    return tf_transform_output.transformed_feature_spec().copy()


def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=64):
    """Creates an input function reading from transformed data.

    Uses data_accessor to properly read compressed TFRecord files
    produced by the TFX Transform component.

    Args:
        file_pattern: List of file patterns for the input data.
        data_accessor: DataAccessor for reading data.
        tf_transform_output: The TFTransformOutput object.
        batch_size: Batch size for the dataset.

    Returns:
        A tf.data.Dataset of (features, labels) tuples.
    """
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=transformed_name(LABEL_KEY),
        ),
        tf_transform_output.transformed_metadata.schema,
    ).repeat()


def _get_feature_columns():
    """Build feature column lists for the model.

    Returns:
        A list of feature column names for the model input.
    """
    feature_columns = []

    for feature in NUMERICAL_FEATURES:
        feature_columns.append(transformed_name(feature))

    for feature in CATEGORICAL_FEATURES:
        feature_columns.append(transformed_name(feature))

    return feature_columns


def _build_keras_model():
    """Build a Keras DNN model for binary classification.

    Architecture: Dense(256) -> Dense(128) -> Dense(64) -> Dense(1)
    with BatchNormalization and Dropout between layers.

    Args:
        feature_columns: List of feature column names.

    Returns:
        A compiled Keras model.
    """
    input_features = []
    concat_features = []
    input_layers = {}

    for feature in NUMERICAL_FEATURES:
        name = transformed_name(feature)
        inp = layers.Input(shape=(1,), name=name, dtype=tf.float32)
        input_features.append(inp)
        concat_features.append(inp)
        input_layers[name] = inp

    for feature in CATEGORICAL_FEATURES:
        name = transformed_name(feature)
        inp = layers.Input(shape=(1,), name=name, dtype=tf.int64)
        input_features.append(inp)
        # Cast int64 to float32 before concatenation
        casted = tf.cast(inp, tf.float32)
        concat_features.append(casted)
        input_layers[name] = inp

    # Concatenate all inputs (all float32 now)
    x = layers.Concatenate()(concat_features)

    # Hidden layers with BatchNorm and Dropout
    for units in HIDDEN_UNITS:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(DROPOUT_RATE)(x)

    # Output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)

    # pylint: disable=no-member
    model = tf.keras.Model(inputs=input_features, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
            tf.keras.metrics.AUC(name='auc'),
        ]
    )
    # pylint: enable=no-member

    model.summary()

    return model


def run_fn(fn_args):
    """Train the model using the Trainer component.

    This function is called by the TFX Trainer component. It builds a DNN
    classifier, trains it on the transformed data, and exports a SavedModel
    with serving signatures.

    Args:
        fn_args: FnArgs object containing training arguments.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Create training and evaluation datasets
    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=TRAIN_BATCH_SIZE,
    )

    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        tf_transform_output,
        batch_size=EVAL_BATCH_SIZE,
    )

    # Build the model
    model = _build_keras_model()

    # Train the model
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=10,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(  # pylint: disable=no-member
                monitor='val_binary_accuracy',
                patience=5,
                restore_best_weights=True
            ),
        ]
    )

    # Define serving signatures
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
        ),
    }

    # Save the model
    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures=signatures,
    )
