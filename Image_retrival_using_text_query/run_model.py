import os
import collections
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from absl import flags
from data_loader import write_data
from absl import logging
from absl import app
from model_architecture import project_embeddings, create_vision_encoder, create_text_encoder

FLAGS = flags.FLASG

flags.DEFINE_integer(
    'training_samples', 40.000,
    'size of training dataset MS-COCO 80K')

flags.DEFINE_integer(
    'val_samples', 5000,
    'size of validation dataset MS-COCO val or sperate from Train data.')

flags.DEFINE_integer(
    'images_per_file', 2000,
    'Splitting train folder each subset train folders.')

flags.DEFINE_float(
    'num_captions', 2,
    'Number of captions will make training sample increasing X times')


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')


# Pre-Training and Finetune
if __name__ == '__main__':

    app.run(main)
