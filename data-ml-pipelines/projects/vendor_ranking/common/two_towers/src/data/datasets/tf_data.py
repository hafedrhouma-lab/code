import tensorflow as tf
import pandas as pd


def create_tf_dataset_from_tensor_slices(input_df: pd.DataFrame) -> tf.data.Dataset:
    input_dict = {col: input_df[col].values for col in input_df.columns}
    tf_dataset = tf.data.Dataset.from_tensor_slices(input_dict)
    tf_dataset = tf_dataset.map(
        lambda features: {
            col: tf.convert_to_tensor(features[col])
            for col in input_df.columns
        }
    )
    return tf_dataset


def create_training_data(positive_data, negative_data, batch_size, ns_ratio):
    positive_tf_dataset = create_tf_dataset_from_tensor_slices(positive_data).shuffle(
        positive_data.shape[0],
        reshuffle_each_iteration=True
    ).batch(batch_size).cache()

    negative_tf_dataset = create_tf_dataset_from_tensor_slices(negative_data)

    batch_size_negative = min(
        negative_data.shape[0],
        batch_size * ns_ratio
    )

    negative_tf_dataset = negative_tf_dataset.shuffle(
        negative_data.shape[0],
        reshuffle_each_iteration=True
    ).repeat().batch(
        int(batch_size_negative),
        drop_remainder=True
    )
    train_ds = tf.data.Dataset.zip(
        (positive_tf_dataset, negative_tf_dataset)
    )

    return train_ds
