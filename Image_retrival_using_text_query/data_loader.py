import tensorflow as tf
import os 
import json
import collections
import numpy as np
from absl import flags
from absl import logging

FLAGS= flags.FLAGS

#***************************************************
# Section Dataset already Download
#***************************************************
root_dir = "/shared_SSD_20TB/SSL-TEAM/Rick/"
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
if not os.path.exists(image_dir): 
    image_zip = tf.keras.utils.get_file("train2014.zip", 
                    cache_dir= os.path.abspath("."), 
                    origin= "http://images.cocodataset.org/zips/train2014.zip",
                    extract= True,  
                    )
    os.remove(image_zip)

## Print out the data is Download and Extracted Successfully
print("Dataset is downloaded and extracted successfully.")

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

#***************************************************
#Process and Save data to TFRecord files
#***************************************************
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

def bytes_feature(value): 
    return tf.train.Feature(bytes_list= tf.train.ByteList(value=[value]))

def create_example(image_path, caption): 
    feature={
        "caption": bytes_feature(caption.encode()), 
        "raw_image": bytes_feature(tf.io.read_file(image_path).numpy()), 
    }
    return tf.train.Example(features= tf.train.Features(feature= feature))

def write_tfrecords(file_name, image_paths):
    caption_list =[]
    image_path_list=[]

    for image_path in image_paths: 
        captions = image_path_to_caption[image_path][:captions_per_image]
        caption_list.extend(captions)
        image_path_list.extend([image_path] * len(captions))
    with tf.io.TFRecordWriter(file_name) as writer:
        for example_idx  in range(len(image_path_list)): 
            example= create_example(image_path_list[example_idx], caption_list[example_idx])
            writer.write(example.SerializeToString)
    
    return example_idx + 1

def write_data(image_paths, num_files, files_prefix):
    example_counter = 0
    for file_idx in tqdm(range(num_files)):
        file_name = files_prefix + "-%02d.tfrecord" % (file_idx)
        start_idx = images_per_file * file_idx
        end_idx = start_idx + images_per_file
        example_counter += write_tfrecords(file_name, image_paths[start_idx:end_idx])
    return example_counter

