from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import numpy as np
import tensorflow as tf

from modules.evaluations import get_val_data, perform_val
from modules.models import ArcFaceModel
from modules.utils import set_memory_growth, load_yaml, l2_norm
import modules.dataset as dataset


flags.DEFINE_string('cfg_path', './configs/arc_res50.yaml', 'config file path')
flags.DEFINE_string('gpu', '0', 'which gpu to use')
flags.DEFINE_string('img_path', '', 'path to input image')

def parse_tfrecord_creator():
    def parse_tfrecord(tfrecord):
        features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                    'image/filename': tf.io.FixedLenFeature([], tf.string),
                    'image/encoded': tf.io.FixedLenFeature([], tf.string)}
        x = tf.io.parse_single_example(tfrecord, features)
        x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
        y_train = tf.cast(x['image/source_id'], tf.float32)
        return (x_train, y_train), y_train
    return parse_tfrecord


def main(_argv):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(FLAGS.cfg_path)

    model = ArcFaceModel(size=cfg['input_size'],
                         backbone_type=cfg['backbone_type'],
                         num_classes=cfg['num_classes'],
                         head_type=cfg['head_type'],
                         embd_shape=cfg['embd_shape'],
                         w_decay=cfg['w_decay'],
                         training=False)

    ckpt_path = tf.train.latest_checkpoint('./checkpoints/' + cfg['sub_name'])
    if ckpt_path is not None:
        print("[*] load ckpt from {}".format(ckpt_path))
        model.load_weights(ckpt_path)
    else:
        print("[*] Cannot find ckpt from {}.".format(ckpt_path))
        exit()
    if cfg['train_dataset']:
        logging.info("load ms1m dataset.")
        dataset_len = cfg['num_samples']
        

        raw_dataset = tf.data.TFRecordDataset(cfg['train_dataset'])
        dset = raw_dataset.map(parse_tfrecord_creator())
        dset = dset.batch(1)

        for idx, parsed_record in enumerate(dset.take(1)):
            (x_train, _), y_train = parsed_record
            print('x_train:', x_train.shape)
            print('x_train:', x_train)
            print('y_train:', y_train)
            prediction = model.predict(x_train)
            print('prediction:', prediction.shape)
            print(prediction)
if __name__ == '__main__':
    app.run(main)
   
