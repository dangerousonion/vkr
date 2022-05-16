import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.applications import ResNet50, MobileNetV2
import numpy as np
from absl import logging


class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x):
        return super().call(x, tf.constant(False))
 
def load_model(MNV2):
    x = inputs = Input([112, 112, 3], name='input_image')
    if MNV2:
        x = MobileNetV2(input_shape=[112, 112, 3], include_top=False, weights='imagenet')(x)
    else:
        x = ResNet50(input_shape=[112, 112, 3], include_top=False, weights='imagenet')(x)
        
    x1 = inputs1 = Input(x.shape[1:])
    x1 = BatchNormalization()(x1)
    x1 = Dropout(rate=0.5)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x1)
    x1 = BatchNormalization()(x1)
    embds = Model(inputs1, x1, name='OutputLayer')(x)
    model = Model(inputs, embds, name='arcface_model')
    ckpt_path = tf.train.latest_checkpoint('/kaggle/working/checkpoints/arc_mbv2' if MNV2 else '/kaggle/working/checkpoints/training')
    if ckpt_path is not None:
        print("Load weights from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
    else:
        print("Cannot find checkpoint from {}.".format(ckpt_path))
        exit()
    return model



def init_tf(gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                #logging.info("Detect {} Physical GPUs, {} Logical GPUs.".format( len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.info(e)

def parse_tfrecord(tfrecord):
    features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                'image/filename': tf.io.FixedLenFeature([], tf.string),
                'image/encoded': tf.io.FixedLenFeature([], tf.string)}
    x = tf.io.parse_single_example(tfrecord, features)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)/255
    y_train = tf.cast(x['image/source_id'], tf.float32)
    return x_train, y_train
    

def dataset_dimensions(dset):
    num_samples = 0
    max_class_id = 0    
    for (_, y) in dset:
        num_samples += 1
        class_id = tf.cast(y, tf.int32).numpy()
        if class_id > max_class_id:
            max_class_id = class_id
    return num_samples,  max_class_id + 1

def predict(dset, model, sample_vectors, sample_classes, do_flip):
    total_vol = 0
    print('predict:', end = ' ')  
    dset = dset.batch(200)
    for (x, y) in dset:
        batch_size = y.shape[0]
        if do_flip:
            prediction = model(x,  training=False) + model(x[:, :, ::-1, :],  training=False)
        else:
            prediction = model(x,  training=False)
        sample_vectors[total_vol: total_vol + batch_size] = (prediction/np.linalg.norm(prediction, axis=1, keepdims=True)).numpy()
        sample_classes[total_vol: total_vol + batch_size] = tf.cast(y, tf.int32).numpy()
        total_vol += batch_size
        print('.', end = '', flush = True)    
    print()
    return total_vol
            