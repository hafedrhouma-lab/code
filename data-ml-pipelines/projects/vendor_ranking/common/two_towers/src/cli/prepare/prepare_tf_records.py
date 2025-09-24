import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import logging
import pickle
import tensorflow as tf

from . import get_feature_description

logger = tf.get_logger()
logger.setLevel(logging.ERROR)


def flatten_and_suffix(pos_batch, neg_batch):
    flattened_batch = {}
    for key, value in pos_batch.items():
        flattened_batch[f'{key}_pos'] = value
    for key, value in neg_batch.items():
        flattened_batch[f'{key}_neg'] = value
    return flattened_batch


def create_example(features, feature_description):
    feature_dict = {}
    dtype_to_feature = {
        tf.float32: lambda v: tf.train.Feature(float_list=tf.train.FloatList(value=v)),
        tf.int64: lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=v)),
        tf.string: lambda v: tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    }

    for key, feature_type in feature_description.items():
        try:
            dtype = feature_type.dtype
            feature_value = features[key].numpy()

            if dtype in dtype_to_feature:
                feature_dict[key] = dtype_to_feature.get(dtype)(feature_value)
            else:
                raise ValueError(f"Unsupported feature type for {key}")

        except Exception as e:
            logger.error(f"Error processing feature {key}: {e}")
            return None

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example.SerializeToString()


class PrepareTFRecords:
    def __init__(
            self,
            train_ds: tf.data.Dataset,
            positive_data_size: int,
            negative_data_size: int,
            params: dict,
            feature_description_file: str,
            tfrecord_file: str
    ):

        self.feature_description_file = feature_description_file
        self.tfrecord_file = tfrecord_file

        self.train_ds = train_ds
        self.positive_data_size = positive_data_size
        self.negative_data_size = negative_data_size
        self.params = params

    def generate(self):
        flattened_train_ds = self.train_ds.map(
            lambda pos, neg: flatten_and_suffix(pos, neg)
        )

        batch_size_negative = min(
            self.negative_data_size,
            self.params.get('train_batch') * self.params.get('ns_ratio')
        )

        feature_description = get_feature_description(
            self.params.get('train_batch'),
            int(batch_size_negative)
        )

        with open(self.feature_description_file, 'wb') as f:
            pickle.dump(feature_description, f)

        with tf.io.TFRecordWriter(self.tfrecord_file) as writer:
            # Iterate over the dataset and write all except the last record
            total_records = sum(1 for _ in flattened_train_ds)  # Count total records
            for index, example in enumerate(flattened_train_ds):
                if index < total_records - 1:  # Skip the last record
                    try:
                        serialized_example = create_example(example, feature_description)
                        if serialized_example is not None:
                            writer.write(serialized_example)
                        else:
                            logger.warning(f"Skipped a bad record: {example}")
                    except Exception as e:
                        logger.error(f"Failed to write example at index {index}: {e}")

        logger.info(f"Finished writing TFRecord file: {self.tfrecord_file}")
