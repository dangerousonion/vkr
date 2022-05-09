from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tqdm
import glob
import random
import tensorflow as tf


flags.DEFINE_string('dataset_path', '/kaggle/input/testset/lfw-deepfunneled/',
                    'path to dataset')
flags.DEFINE_string('output_path', '/kaggle/working/data/ms1m_bin.tfrecord',
                    'path to ouput tfrecord')


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(img_str, source_id, filename):
    # Create a dictionary with features that may be relevant.
    feature = {'image/source_id': _int64_feature(source_id),
               'image/filename': _bytes_feature(filename),
               'image/encoded': _bytes_feature(img_str)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main(_):
    dataset_path = FLAGS.dataset_path

    if not os.path.isdir(dataset_path):
        logging.info('Please define valid dataset path.')
    else:
        logging.info('Loading {}'.format(dataset_path))

    samples = []
    logging.info('Reading data list...')
    for id_name in tqdm.tqdm(os.listdir(dataset_path)):
        img_paths = glob.glob(os.path.join(dataset_path, id_name, '*.jpg'))
        for img_path in img_paths:
            filename = os.path.join(id_name, os.path.basename(img_path))
            samples.append((img_path, id_name, filename))
    random.shuffle(samples)

    logging.info('Writing tfrecord file...')
    with tf.io.TFRecordWriter(FLAGS.output_path) as writer:
        for img_path, id_name, filename in tqdm.tqdm(samples):
            tf_example = make_example(img_str=open(img_path, 'rb').read(),
                                      source_id=int(id_name),
                                      filename=str.encode(filename))
            writer.write(tf_example.SerializeToString())


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
