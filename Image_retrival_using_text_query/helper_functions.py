import tensorflow as tf
from absl import flags
FLAGS= flags.FLAGS


#******************************************
#reading caption string and convert to tensors
#******************************************

feature_description = {"caption": tf.io.FixedLenFeature([], tf.string), 
                        "raw_image": tf.io.FixedLenFeature([], tf.string)}

def read_example(example): 
    features= tf.io.parse_single_example(example, feature_description)
    raw_image=features.pop("raw_image")
    features["image"]= tf.resize(tf.decode_jpeg(raw_image, channels=3), size(FLAGS.IMG_WIDTH, FLAGS.IMG_HEIGHT ))
    return features

def get_dataset(file_pattern, batch_size): 
    return (tf.data.TFRecordDataset(tf.data.Dataset.list_files(file_pattern))
    .map(read_example, num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,)
    .shuffle(FLASG.batch_size)