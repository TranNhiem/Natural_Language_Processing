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

    
    #***************************************************
    # Section reading text Json Format file 
    #***************************************************

    with open(annotation_file, "r") as f: 
        annotations = json.load(f)["annotations"]

    image_path_to_caption= collections.defaultdict(list)
    for element in annotations: 
        caption = f"{element['caption].lower().rstrip('.')}"
        image_path= image_dir + "/COCO_train2014_" + "%012d.jpg" % (element["image_id"])
        image_path_to_caption[image_path].append(caption)

    image_paths = list(image_path_to_caption.keys())
    print(f"Number of images: {len(image_path)}")


    train_size = FLAGS.training_samples
    valid_size = FLASG.val_samples
    captions_per_image = FLAGS.num_captions
    # define for devided data into sub train files
    images_per_file = FLAGS.images_per_file
    train_image_path= image_paths[: train_size]
    num_train_files= int(np.ceil(train_size / images_per_file))
    train_files_prefix = os.path.join(tfrecords_dir, "train")

    valid_image_paths = image_paths[-val_size: ]
    num_val_files = int(np.ceil(valid_size / images_per_file))
    valid_files_prefix = os.path.join(tfrecords_dir, "valid")

    tf.io.gfile.makedirs(tfrecords_dir)
    train_example_count = write_data(FLAGS.train_image_paths, FLAGS.num_train_files, train_files_prefix)
    print(f"{train_example_count} training examples were written to tfrecord files.")
    valid_example_count = write_data(FLAGS.valid_image_paths, FLAGS.num_valid_files, valid_files_prefix)
    print(f"{valid_example_count} evaluation examples were written to tfrecord files.")




# Pre-Training and Finetune
if __name__ == '__main__':

    app.run(main)
