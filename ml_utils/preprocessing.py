from typing import TypedDict, List, Dict, Tuple, Union
from keras import layers
import tensorflow as tf


class FeatureDict(TypedDict):
    num_features: List[str]
    cat_features: List[str]


def create_input_layers(feature_dict: FeatureDict) -> Dict[str, layers.Input]:
    """Each feature in feature_dict will get an input layer under the same name.

    Args:
        feature_dict (FeatureDict): {"num_features": list of features, "cat_features": list of features}

    Returns:
        Dict[str, layers.Input]: dictionary of input layers.
    """

    inputs = {}

    for feature in feature_dict["cat_features"]:
        inputs[feature] = layers.Input(name=feature, shape=(), dtype=tf.int32)
    for feature in feature_dict["num_features"]:
        inputs[feature] = layers.Input(name=feature, shape=(), dtype=tf.float64)
    return inputs


def embed_features(
    inputs: dict, vocab_dict: Dict[str, Union[int, list]], output_dim: int, stack_axis=1, concat_axis=1, **kwargs
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Run categorical features through embedding layers. Numerical features are left alone.

    Args:
        inputs (dict): output of create_input_layers().
        vocab_dict (Dict[str, int]): key is feature name, value is size of vocab/list of vocab.
        output_dim (int): output embedding dim.
        stack_axis (int, optional): axis to stack cat_features along. Defaults to 1.
        concat_axis (int, optional): axis to concat num_fetaures along. Defaults to 1.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: (cat_features, num_features)
    """

    num_features = []
    cat_features = []

    for feature in inputs:
        if feature in vocab_dict.keys():
            if isinstance(vocab_dict[feature], int):
                vocab_size = vocab_dict[feature]
                cat_features.append(layers.Embedding(input_dim=vocab_size, output_dim=output_dim, **kwargs)(inputs[feature]))
            elif isinstance(vocab_dict[feature], list):
                vocab_size = len(vocab_dict[feature])
                encoded_features = layers.StringLookup(vocabulary=vocab_dict[feature])(inputs[feature])
                cat_features.append(layers.Embedding(input_dim=vocab_size, output_dim=output_dim, **kwargs)(encoded_features))
        else:
            num_features.append(tf.expand_dims(inputs[feature], -1))
    return tf.stack(cat_features, axis=stack_axis), tf.concat(num_features, axis=concat_axis)
