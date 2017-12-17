"""
读取训练数据集。
"""
import pickle
import numpy as np


def dense_to_one_hot(labels_dense, num_classes=10):
    """
    将标量数据转为one-hot向量。
    """
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense] = 1
    return labels_one_hot


class DataSet(object):
    """
    数据集。
    """
    def __init__(self, images, labels):
        assert images.shape[0] == len(labels)
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0/255.0)
        self._ori_labels = labels
        self._images = images
        self._labels = dense_to_one_hot(labels)
        self._index = 0
        self._nums = images.shape[0]

    def next_batch(self, batch_size):
        assert self._nums >= batch_size > 0
        start = self._index
        end = start + batch_size
        if end > self._nums:
            start = 0
            end = batch_size

            index_shuffle = np.arange(self._nums)
            np.random.shuffle(index_shuffle)
            self._images = self._images[index_shuffle]
            self._labels = self._labels[index_shuffle]

        self._index = end
        return self._images[start:end], self._labels[start:end]

    def get_ori_labels(self):
        return self._ori_labels


class DataSets(object):
    pass


def unpickle(file):
    """
    读取文件中的图像与标签
    """
    with open(file, 'rb') as fo:
        filedata = pickle.load(fo, encoding='bytes')
    data = filedata[b'data']
    labels = filedata[b'labels']
    return data, labels


def read_data_set(data_dir):
    """
    读取目标文件夹下所有测试数据
    """
    data_batch_1 = 'data_batch_1'
    data_batch_2 = 'data_batch_2'
    data_batch_3 = 'data_batch_3'
    data_batch_4 = 'data_batch_4'
    data_batch_5 = 'data_batch_5'
    test_batch = 'test_batch'
    train_1 = unpickle(data_dir + '/' + data_batch_1)
    train_2 = unpickle(data_dir + '/' + data_batch_2)
    train_3 = unpickle(data_dir + '/' + data_batch_3)
    train_4 = unpickle(data_dir + '/' + data_batch_4)
    train_5 = unpickle(data_dir + '/' + data_batch_5)
    test = unpickle(data_dir + '/' + test_batch)

    train_images = np.vstack((train_1[0], train_2[0], train_3[0], train_4[0], train_5[0]))
    train_labels = train_1[1] + train_2[1] + train_3[1] + train_4[1] + train_5[1]

    data_sets = DataSets()
    data_sets.train = DataSet(train_images, np.asarray(train_labels))
    data_sets.test = DataSet(test[0], np.asarray(test[1]))
    return data_sets


