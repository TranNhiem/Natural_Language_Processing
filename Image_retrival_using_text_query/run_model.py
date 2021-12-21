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

FLAGS = flags.FLASG


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')


# Pre-Training and Finetune
if __name__ == '__main__':

    app.run(main)
