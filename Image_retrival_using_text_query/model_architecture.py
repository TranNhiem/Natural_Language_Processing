import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_text as text
import tensorflow_addons as tfa
from tqdm import tqdm
from data_loader import write_data
import tensorflow_hub as hub
from absl import flags

FLASG = flags.FLAGS



'''
Description: Implementation of a dual encoder model for retrieving images that match natural language queries.

The example demonstrates how to build a dual encoder (also known as two-tower) neural network model to search for images using natural language. The model is inspired by the CLIP approach, introduced by Alec Radford et al. The idea is to train a vision encoder and a text encoder jointly to project the representation of images and their captions into the same embedding space, such that the caption embeddings are located near the embeddings of the images they describe.

This example requires TensorFlow 2.4 or higher. In addition, TensorFlow Hub and TensorFlow Text are required for the BERT model, and TensorFlow Addons is required for the AdamW optimizer. These libraries can be installed using the following command:

'''
# **************************************************
# Projection Dimension Embedding (Vision and Text Representation SAME Dim)
# **************************************************


def project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate):
    projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.nn.gelu(projected_embeddings)
        x = layers.Dense(projection_dims)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Add()([projected_embeddings, x])
        projected_embeddings = layers.LayerNormalization()(x)
    return projected_embeddings

# **************************************************
# Vision Encoder Finetune ResNet Conv Encoder
# **************************************************


def create_vision_encoder(num_projection_layers, projection_dims, dropout_rate, trainable=False):
    # Load the pre-trained Xception model to be used as the base encoder.
    resnet = tf.keras.applications.resnet50.ResNet50(
        include_top=False, weights="imagenet", pooling="avg")

    # Set the trainability of the base encoder.
    for layer in resnet.layers:
        layer.trainable = trainable

    # Receive the images as inputs.
    inputs = layers.Input(shape=(224, 224, 3), name="image_input")
    # Preprocess the input image.
    resnet_input = tf.keras.applications.resnet50.preprocess_input(inputs)
    # Generate the embeddings for the images using the xception model.
    embeddings = resnet(resnet_input)
    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate)
    # Create the vision encoder model.
    return tf.keras.Model(inputs, outputs, name="vision_encoder")

# **************************************************
# Text Encoder Fintue BERT Architecture
# **************************************************


def create_text_encoder(num_projection_layers, projection_dims, dropout_rate, trainable=False):
    # Load the BERT preprocessing module.
    preprocess = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2",
        name="text_preprocessing",
    )
    # Load the pre-trained BERT model to be used as the base encoder.
    bert = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1",
        "bert",
    )
    # Set the trainability of the base encoder.
    bert.trainable = trainable
    # Receive the text as inputs.
    inputs = layers.Input(shape=(), dtype=tf.string, name="text_input")
    # Preprocess the text.
    bert_inputs = preprocess(inputs)
    # Generate embeddings for the preprocessed text using the BERT model.
    embeddings = bert(bert_inputs)["pooled_output"]
    # Project the embeddings produced by the model.
    outputs = project_embeddings(
        embeddings, num_projection_layers, projection_dims, dropout_rate
    )
    # Create the text encoder model.
    return tf.keras.Model(inputs, outputs, name="text_encoder")
