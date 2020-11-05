import tensorflow as tf

mapping = {
            "datetime":tf.io.FixedLenFeature([], dtype=tf.int64),
            "temperature":tf.io.FixedLenFeature([], dtype=tf.int64),
            "humidity":tf.io.FixedLenFeature([], dtype=tf.int64),
            "audio":tf.io.FixedLenFeature([], dtype=tf.float32)
        }

def decoder(record):
    return tf.io.parse_single_example(
        record,
        mapping
    )

dataset = tf.data.TFRecordDataset(["fusion.tfrecord"]).map(decoder)

for data in dataset.take(4):
    print(data)