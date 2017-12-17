"""
生成embedding模型
"""
from tensorflow.contrib.tensorboard.plugins import projector
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from cifar10 import input_data

LOG_DIR = 'visualization_logs'
TO_EMBED_COUNT = 1000
PATH_FOR_MODEL = "model.ckpt"
PATH_FOR_SPRITE = 'thumbnails.png'
PATH_FOR_METADATA = 'metadata.tsv'
NAME_TO_VISUALISE_VARIABLE = 'cifar10_embedding'


def create_embedding():
    """
    创建embedding模型。
    """
    cifar10 = input_data.read_data_set("cifar10_data")
    images = cifar10.test.next_batch(TO_EMBED_COUNT)[0]

    # 建立embedding
    embedding_var = tf.Variable(images, name=NAME_TO_VISUALISE_VARIABLE)
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    # 建立embedding projectorc
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = PATH_FOR_METADATA
    embedding.sprite.image_path = PATH_FOR_SPRITE
    embedding.sprite.single_image_dim.extend([32, 32, 3])

    projector.visualize_embeddings(summary_writer, config)

    # 保存
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, PATH_FOR_MODEL), 1)


def vector_to_matrix(images):
    """
    将一系列图片以矩阵方式排列，用来生成精灵图。
    """
    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)

    img_h = images.shape[1]
    img_w = images.shape[2]
    img_d = images.shape[3]
    side = int(np.ceil(np.sqrt(images.shape[0])))
    sprite_image = np.ones((img_h * side, img_w * side, img_d))

    for i in range(side):
        for j in range(side):
            position = i * side + j
            if position < images.shape[0]:
                sprite_image[
                i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w
                ] = images[position]
    return sprite_image


def produce_sprite_image():
    """
    生成精灵图。
    """
    cifar10 = input_data.read_data_set("cifar10_data")
    images = cifar10.test.next_batch(TO_EMBED_COUNT)[0]
    images = vector_to_matrix(images)
    plt.imsave(os.path.join(LOG_DIR, PATH_FOR_SPRITE), images)


def produce_metadata():
    """
    生成元数据
    """
    cifar10 = input_data.read_data_set("cifar10_data")
    labels = cifar10.test.get_ori_labels()[0:TO_EMBED_COUNT]
    np.savetxt(os.path.join(LOG_DIR, PATH_FOR_METADATA), labels, fmt='%d', delimiter='\n')


if __name__ == '__main__':
    # produce_metadata()
    # produce_sprite_image()
    create_embedding()
