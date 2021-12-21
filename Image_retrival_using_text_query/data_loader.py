import tensorflow as tf
import os 
import json
import collections
import numpy as np
from absl import flags
from absl import logging
from tqdm import tqdm

FLAGS= flags.FLAGS

#***************************************************
# Section Dataset already Download
#***************************************************
root_dir = FLAGS.root_dir
annotations_dir = os.path.join(root_dir, "annotations")
images_dir = os.path.join(root_dir, "train2014")
# Prepare for the tfrecord format 
tfrecords_dir = os.path.join(root_dir, "tfrecords")
annotation_file = os.path.join(annotations_dir, "captions_train2014.json")

#***************************************************
# Section Dataset is not existing 
#***************************************************

## if annotation caption is not available
if not os.path.exists(annotations_dir): 
    annotation_zip = tf.keras.utils.get_file("captions.zip",cache_dir = os.path.abspath("."), 
                            origin= "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
                            extract = True, 
                            )
    os.remove(annotation_zip)

## if Image files is not available 
if not os.path.exists(images_dir): 
    image_zip = tf.keras.utils.get_file("train2014.zip", 
                    cache_dir= os.path.abspath("."), 
                    origin= "http://images.cocodataset.org/zips/train2014.zip",
                    extract= True,  
                    )
    os.remove(image_zip)

## Print out the data is Download and Extracted Successfully
print("Dataset is downloaded and extracted successfully.")


#***************************************************
#Process and Save data to TFRecord files
#***************************************************


def bytes_feature(value): 
    return tf.train.Feature(bytes_list= tf.train.ByteList(value=[value]))

def create_example(image_path, caption): 
    feature={
        "caption": bytes_feature(caption.encode()), 
        "raw_image": bytes_feature(tf.io.read_file(image_path).numpy()), 
    }
    return tf.train.Example(features= tf.train.Features(feature= feature))

def write_tfrecords(file_name, image_paths, image_path_to_caption):
    caption_list =[]
    image_path_list=[]

    for image_path in image_paths: 
        captions = image_path_to_caption[image_path][:FLAGS.captions_per_image]
        caption_list.extend(captions)
        image_path_list.extend([image_path] * len(captions))

    with tf.io.TFRecordWriter(file_name) as writer:
        for example_idx  in range(len(image_path_list)): 
            example= create_example(image_path_list[example_idx], caption_list[example_idx])
            writer.write(example.SerializeToString)
    
    return example_idx + 1

def write_data(image_paths, num_files, files_prefix, images_per_file, ):
    example_counter = 0
    for file_idx in tqdm(range(num_files)):
        file_name = files_prefix + "-%02d.tfrecord" % (file_idx)
        start_idx = images_per_file * file_idx
        end_idx = start_idx + images_per_file
        example_counter += write_tfrecords(file_name, image_paths[start_idx:end_idx], image_path_to_caption, )
    return example_counter


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
    .shuffle(FLAGS.train_batch_size*10)
    .prefetch(buffer_size= tf.data.AUTOTUNE)
    .batch(batch_size)
    )