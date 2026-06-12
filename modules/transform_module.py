"""Transform module for Credit Risk Prediction TFX Pipeline.

This module contains the preprocessing_fn function used by the TFX Transform
component to perform feature engineering on the credit risk dataset.
"""

import tensorflow as tf
import tensorflow_transform as tft


# Feature keys
NUMERICAL_FEATURES = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
]

CATEGORICAL_FEATURES = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file',
]

LABEL_KEY = 'loan_status'

# Transformed feature name helpers
def transformed_name(key):
    """Generate the name for a transformed feature.

    Args:
        key: The original feature key.

    Returns:
        The transformed feature name with '_xf' suffix.
    """
    return key + '_xf'


def preprocessing_fn(inputs):
    """Preprocess input features into transformed features.

    This function defines the preprocessing pipeline for the credit risk dataset.
    Numerical features are normalized using z-score scaling.
    Categorical features are encoded using vocabulary lookup.

    Args:
        inputs: A dict of feature keys to raw feature tensors.

    Returns:
        A dict of feature keys to transformed feature tensors.
    """
    outputs = {}

    # Scale numerical features using z-score normalization
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(
            _fill_missing(inputs[feature])
        )

    # Encode categorical features using vocabulary
    for feature in CATEGORICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.compute_and_apply_vocabulary(
            _fill_missing_string(inputs[feature]),
            top_k=10,
            num_oov_buckets=1
        )

    # Keep the label as-is
    outputs[transformed_name(LABEL_KEY)] = _fill_missing(inputs[LABEL_KEY])

    return outputs


def _fill_missing(x):
    """Replace missing values in a SparseTensor with 0.

    Args:
        x: A SparseTensor or dense Tensor.

    Returns:
        A dense Tensor with missing values filled.
    """
    if isinstance(x, tf.SparseTensor):
        default_value = 0
        dense = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value
        )
    else:
        dense = x

    return tf.squeeze(dense, axis=1)


def _fill_missing_string(x):
    """Replace missing values in a string SparseTensor with empty string.

    Args:
        x: A SparseTensor or dense Tensor of strings.

    Returns:
        A dense Tensor with missing values filled.
    """
    if isinstance(x, tf.SparseTensor):
        default_value = ''
        dense = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value
        )
    else:
        dense = x

    return tf.squeeze(dense, axis=1)
