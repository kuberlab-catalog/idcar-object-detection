import argparse
import glob
import io
from os import path

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

import dataset_util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', help='Output directory for dataset')

    parser.add_argument('--train-images', help='Path to train images dir')
    parser.add_argument('--test-images', help='Path to test images dir')
    parser.add_argument('--train-labels', help='Path to train labels dir')
    parser.add_argument('--test-labels', help='Path to test labels dir')

    return parser.parse_args()


def create_tf_example(example):
    image_path = example['image_filename']
    label_path = example['label_filename']

    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)

    height = image.size[1]  # Image height
    width = image.size[0]  # Image width
    # Filename of the image. Empty if image is not from file
    filename = path.basename(image_path).encode('utf8')
    # Encoded image bytes
    encoded_image_data = encoded_jpg
    # b'jpeg' or b'png'
    image_format = image.format.lower().encode('utf8')

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    # Parse labels
    # each line in format 27,17,103,22,106,47,30,45,Please
    # xmin,ymin,xmax,ymax

    data = pd.read_csv(label_path, header=None, error_bad_lines=False)
    polyx = data.iloc[:, [0, 2, 4, 6]].values
    polyy = data.iloc[:, [1, 3, 5, 7]].values
    xmin = np.min(polyx, axis=1)
    xmax = np.max(polyx, axis=1)
    ymin = np.min(polyy, axis=1)
    ymax = np.max(polyy, axis=1)
    labels = data.iloc[:, 8].values

    for i, label in enumerate(labels):
        if label == '###':
            continue

        xmins.append(float(xmin[i]) / width)
        xmaxs.append(float(xmax[i]) / width)
        ymins.append(float(ymin[i]) / height)
        ymaxs.append(float(ymax[i]) / height)
        classes_text.append('Text'.encode('utf-8'))
        classes.append(1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def write_record(image_dir, label_dir, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    image_paths = glob.glob(path.join(image_dir, '*'))

    added = 0
    for image_path in image_paths:
        base_image = path.basename(image_path)
        label_base = 'gt_{}.txt'.format(base_image[:base_image.rfind('.')])

        label_path = path.join(label_dir, label_base)

        if not path.exists(label_path):
            tf.logging.info(
                'Skip image path {}, label path {} does not exist.'.format(
                    image_path, label_path
                )
            )
            continue

        tf.logging.info('Adding image {}'.format(image_path))
        example_dict = {
            'image_filename': image_path,
            'label_filename': label_path,
        }

        try:
            tf_example = create_tf_example(example_dict)
            writer.write(tf_example.SerializeToString())
            added += 1
        except Exception as e:
            tf.logging.warning('Skip: {}'.format(str(e)))

    tf.logging.info('Successfully added {} images to dataset.'.format(added))
    tf.logging.info('Writing to {}.'.format(output_file))
    writer.close()


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()

    train_images_dir = args.train_images
    test_images_dir = args.test_images
    train_labels_dir = args.train_labels
    test_labels_dir = args.test_labels

    write_record(train_images_dir, train_labels_dir, path.join(args.output_dir, 'train.record'))
    tf.logging.info('\n\n')
    write_record(test_images_dir, test_labels_dir, path.join(args.output_dir, 'test.record'))


if __name__ == '__main__':
    main()
