import tensorflow as tf


def read_tf_records(
        feature_description,
        tfrecord_file: str,
        number_batch: int = 14000 #FIXME: 14000 should be fixed and always pass number_batch with no default val
):
    # with open(feature_description_file, 'rb') as f:
    #     loaded_feature_description = pickle.load(f)
    def parse_example(example_proto):
        return tf.io.parse_single_example(
            example_proto,
            feature_description
        )

    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

    parsed_dataset = (
        raw_dataset
            .map(parse_example)
            .shuffle(number_batch, reshuffle_each_iteration=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return parsed_dataset
