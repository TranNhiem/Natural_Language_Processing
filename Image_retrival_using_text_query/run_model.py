import os
import json
import numpy as np
import collections
from absl import app
from absl import flags
import tensorflow as tf
from absl import logging
from data_loader import write_data
from tensorflow.keras import layers
from model_architecture import project_embeddings, create_vision_encoder, create_text_encoder

# ************************************************
# Devices Configures
# ************************************************
from config.config  import read_cfg
from config.absl_mock import Mock_Flag

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0:8], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

read_cfg()
flag = Mock_Flag()
FLAGS = flag.FLAGS

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    train_example_count = write_data(FLAGS.train_image_paths, FLAGS.num_train_files, FLAGS.train_files_prefix)
    print(f"{train_example_count} training examples were written to tfrecord files.")
    valid_example_count = write_data(FLAGS.valid_image_paths, FLAGS.num_valid_files, FLAGS.valid_files_prefix)
    print(f"{valid_example_count} evaluation examples were written to tfrecord files.")


# Pre-Training and Finetune
if __name__ == '__main__':

    app.run(main)
