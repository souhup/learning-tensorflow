"""
神经网络模型。
"""
import numpy as np
from PIL import Image
import tensorflow as tf
from cifar10 import input_data


class Rec(object):
    """
    识别图片的卷积神经网络模型。
    """
    def __init__(self):
        with tf.name_scope('input_layer'):
            self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
            self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[])
            self.is_test = tf.placeholder(dtype=tf.bool, shape=[])
            self.labels = tf.placeholder(tf.float32, [None, 10], name='labels')
            self.images = tf.placeholder(tf.float32, [None, 32 * 32 * 3], name='images')
        # self.train()
        # self.calculate_accuracy()

    def weight_variable(self, shape):
        """
        权重初始化函数。
        """
        with tf.name_scope('weight'):
            initial = tf.truncated_normal(shape, stddev=0.05)
            res = tf.Variable(initial, name='weight')
        return res

    def bias_variable(self, shape):
        """
        偏移初始化。
        """
        with tf.name_scope('bias'):
            initial = tf.constant(0.0, shape=shape)
            res = tf.Variable(initial, name='bias')
        return res

    def activate(self, x):
        """
        激活函数。
        """
        with tf.name_scope('activation'):
            res = tf.nn.relu(x)
        return res

    def conv2d(self, x, w):
        """
        卷积函数。
        """
        with tf.name_scope('convolution'):
            res = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        return res

    def max_pool_2x2(self, x):
        """
        池化函数。
        """
        with tf.name_scope('pool'):
            res = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return res

    def distort_image(self, train_image):
        """
        随机处理图片。
        """
        # 随机水平翻转
        train_image = tf.image.random_flip_left_right(train_image)
        # 随机调整亮度
        train_image = tf.image.random_brightness(train_image, max_delta=0.4)
        # 随机调整对比度
        train_image = tf.image.random_contrast(train_image, lower=0.4, upper=1.6)
        # 随机调整色相
        train_image = tf.image.random_hue(train_image, 0.5)
        # 随机调整饱和度
        train_image = tf.image.random_saturation(train_image, lower=0, upper=5)
        return train_image

    def distort_images(self, images, batch_size):
        """
        随机处理批量图片
        """
        images = tf.unstack(images, axis=0, num=batch_size)
        return tf.stack([self.distort_image(image) for image in images], axis=0)

    def create_module(self, batch_size):
        """
        创建模型
        """
        # 第一层的卷积层、激活层、池化层
        with tf.name_scope('first_conv_layer'):
            images = tf.reshape(self.images, [-1, 3, 32, 32])
            images = tf.transpose(images, [0, 2, 3, 1])

            images = tf.cond(self.is_test, lambda: images, lambda: self.distort_images(images, batch_size))

            w_conv1 = self.weight_variable([3, 3, 3, 32])
            b_conv1 = self.bias_variable([32])
            h_conv1 = self.conv2d(images, w_conv1) + b_conv1
            h_act1 = self.activate(h_conv1)

            w_conv1 = self.weight_variable([3, 3, 32, 48])
            b_conv1 = self.bias_variable([48])
            h_conv1 = self.conv2d(h_act1, w_conv1) + b_conv1
            h_act1 = self.activate(h_conv1)

            w_conv1 = self.weight_variable([3, 3, 48, 64])
            b_conv1 = self.bias_variable([64])
            h_conv1 = self.conv2d(h_act1, w_conv1) + b_conv1
            h_act1 = self.activate(h_conv1)

            h_pool1 = self.max_pool_2x2(h_act1)

        # 第二层的卷积层、激活层、池化层
        with tf.name_scope('second_conv_layer'):
            w_conv2 = self.weight_variable([3, 3, 64, 80])
            b_conv2 = self.bias_variable([80])
            h_conv2 = self.conv2d(h_pool1, w_conv2) + b_conv2
            h_act2 = self.activate(h_conv2)

            w_conv2 = self.weight_variable([3, 3, 80, 96])
            b_conv2 = self.bias_variable([96])
            h_conv2 = self.conv2d(h_act2, w_conv2) + b_conv2
            h_act2 = self.activate(h_conv2)

            w_conv2 = self.weight_variable([3, 3, 96, 112])
            b_conv2 = self.bias_variable([112])
            h_conv2 = self.conv2d(h_act2, w_conv2) + b_conv2
            h_act2 = self.activate(h_conv2)

            h_pool2 = self.max_pool_2x2(h_act2)

        # 第三层的卷积层、激活层、池化层
        with tf.name_scope('third_conv_layer'):
            w_conv3 = self.weight_variable([3, 3, 112, 128])
            b_conv3 = self.bias_variable([128])
            h_conv3 = self.conv2d(h_pool2, w_conv3) + b_conv3
            h_act3 = self.activate(h_conv3)

            w_conv3 = self.weight_variable([3, 3, 128, 128])
            b_conv3 = self.bias_variable([128])
            h_conv3 = self.conv2d(h_act3, w_conv3) + b_conv3
            h_act3 = self.activate(h_conv3)

            w_conv3 = self.weight_variable([3, 3, 128, 128])
            b_conv3 = self.bias_variable([128])
            h_conv3 = self.conv2d(h_act3, w_conv3) + b_conv3
            h_act3 = self.activate(h_conv3)

            h_pool3 = self.max_pool_2x2(h_act3)

        # 全连接层
        with tf.name_scope('full_connect_layer'):
            w_fc = self.weight_variable([128 * 4 * 4, 512])
            b_fc = self.bias_variable([512])
            h_pool3_flat = tf.reshape(h_pool3, [-1, 128 * 4 * 4])
            h_fc = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc) + b_fc)

        # 输出层
        with tf.name_scope('output_layer'):
            w_out = self.weight_variable([512, 10])
            b_out = self.bias_variable([10])
            output = tf.nn.softmax(tf.matmul(h_fc, w_out) + b_out)

        # 训练模型
        with tf.name_scope('train_layer'):
            with tf.name_scope('loss'):
                cross_entropy = -tf.reduce_sum(self.labels * tf.log(output + 1e-10)) \
                                + tf.nn.l2_loss(w_conv1) \
                                + tf.nn.l2_loss(w_conv2) \
                                + tf.nn.l2_loss(w_conv3) \
                                + tf.nn.l2_loss(w_out)
                tf.summary.scalar('loss', cross_entropy)
                tf.summary.histogram('w_conv1', w_conv1)
                tf.summary.histogram('w_conv2', w_conv2)
                tf.summary.histogram('w_conv3', w_conv3)
                tf.summary.histogram('w_fc1', w_fc)
                tf.summary.histogram('w_out', w_out)

            with tf.name_scope('accuracy_train'):
                correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(self.labels, 1))
                accuracy_train = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy_train', accuracy_train)

            with tf.name_scope('train'):
                loss = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)
                tf.summary.scalar('learning_rate', self.learning_rate)

        return accuracy_train, loss, output

    def train(self):
        """
        训练函数
        """
        step = 10001
        batch_size = 128
        learning_rate = 1e-3
        cifar10 = input_data.read_data_set("cifar10_data")
        _, loss, _ = self.create_module(batch_size)

        sess = tf.Session()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test', sess.graph)
        sess.run(tf.global_variables_initializer())

        for step in range(step):
            train_batch = cifar10.train.next_batch(batch_size)
            test_batch = cifar10.test.next_batch(batch_size)
            sess.run(loss, feed_dict={
                self.images: train_batch[0],
                self.labels: train_batch[1],
                self.learning_rate: learning_rate,
                self.is_test: False,
                self.keep_prob: 0.5
            })
            if step % 10 == 0:
                print(step)
            if step % 50 == 0 and step > 0:
                train_record = sess.run(merged, feed_dict={
                    self.images: train_batch[0],
                    self.labels: train_batch[1],
                    self.learning_rate: learning_rate,
                    self.is_test: False,
                    self.keep_prob: 0.5
                })
                test_record = sess.run(merged, feed_dict={
                    self.images: test_batch[0],
                    self.labels: test_batch[1],
                    self.learning_rate: learning_rate,
                    self.is_test: True,
                    self.keep_prob: 1
                })
                train_writer.add_summary(train_record, step)
                test_writer.add_summary(test_record, step)
        saver = tf.train.Saver()
        saver.save(sess, "save/session.ckpt")

    def calculate_accuracy(self):
        """
        计算最终准确率
        """
        acc = 0.0
        cifar10 = input_data.read_data_set("cifar10_data")

        sess = tf.Session()
        accuracy, _, _ = self.create_module(1)
        saver = tf.train.Saver()
        saver.restore(sess, "save/session.ckpt")
        for i in range(500):
            test_batch = cifar10.test.next_batch(100)
            acc += sess.run(accuracy, feed_dict={
                self.images: test_batch[0],
                self.labels: test_batch[1],
                self.is_test: True,
                self.keep_prob: 1
            })

        print(acc / 500.0)

    def prepare_detect(self):
        """
        识别图片前的处理。
        """
        self.sess = tf.Session()
        _, _, self.output = self.create_module(1)
        saver = tf.train.Saver()
        saver.restore(self.sess, "save/session.ckpt")
        self.kinds = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }

    def detect_image(self, path):
        """
        检测图片
        """
        img = self.read_img(path)
        result = self.sess.run(self.output, feed_dict={
            self.images: img,
            self.is_test: True,
        })
        return self.kinds[result.argmax()]

    def read_img(self, path):
        """
        读取图片
        """
        img = Image.open(path).resize((32, 32), Image.ANTIALIAS)
        data = np.array(img)
        img.close()
        data = np.multiply(data, 1.0 / 255)
        data = data.transpose(2, 0, 1)
        data = data.reshape(1, 3 * 32 * 32)
        return data
