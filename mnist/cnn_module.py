"""
神经网络文件。
"""
import tensorflow as tf
from mnist import input_data


class Rec(object):
    """
    卷及神经网络识别手写数字。
    """

    def __init__(self):
        # self.input_all_data()
        self.create_module()
        self.restore_module()
        # self.train()

    def weight_variable(self, shape):
        """
        权重初始化函数
        """
        with tf.name_scope('weight'):
            initial = tf.truncated_normal(shape, stddev=0.1)
            res = tf.Variable(initial, name='weight')
        return res

    def bias_variable(self, shape):
        """
        偏移初始化
        """
        with tf.name_scope('bias'):
            initial = tf.constant(0.1, shape=shape)
            res = tf.Variable(initial, name='bias')
        return res

    def activate(self, x):
        """
        激活函数
        """
        with tf.name_scope('activation'):
            res = tf.nn.relu(x)
        return res

    def conv2d(self, x, w):
        """
        卷积函数
        """
        with tf.name_scope('convolution'):
            res = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        return res

    def max_pool_2x2(self, x):
        """
        池化函数
        """
        with tf.name_scope('pool'):
            res = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        return res

    def input_all_data(self):
        """
        读取训练与测试数据
        """
        self.mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

    def create_module(self):
        """
        创建模型
        """

        # 输入数据层
        with tf.name_scope('input'):
            self.images = tf.placeholder(tf.float32, [None, 784], name='images')
            self.labels = tf.placeholder(tf.float32, [None, 10], name='labes')

        # 第一层的卷积层、激活层、池化层
        with tf.name_scope('first_layer'):
            w_conv1 = self.weight_variable([5, 5, 1, 32])
            b_conv1 = self.bias_variable([32])
            x_image = tf.reshape(self.images, [-1, 28, 28, 1])
            h_conv1 = self.conv2d(x_image, w_conv1) + b_conv1
            h_act1 = self.activate(h_conv1)
            h_pool1 = self.max_pool_2x2(h_act1)

        # 第二层的卷积层、激活层、池化层
        with tf.name_scope('second_layer'):
            w_conv2 = self.weight_variable([5, 5, 32, 64])
            b_conv2 = self.bias_variable([64])
            h_conv2 = self.conv2d(h_pool1, w_conv2) + b_conv2
            h_act2 = self.activate(h_conv2)
            h_pool2 = self.max_pool_2x2(h_act2)

        # 全连接层
        with tf.name_scope('full_connect_layer'):
            w_fc1 = self.weight_variable([7 * 7 * 64, 1024])
            b_fc1 = self.bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

        # 输出层
        with tf.name_scope('softmax_layer'):
            w_fc2 = self.weight_variable([1024, 10])
            b_fc2 = self.bias_variable([10])
            self.output = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)

        # 训练模型
        with tf.name_scope('train_layer'):
            with tf.name_scope('loss'):
                cross_entropy = -tf.reduce_sum(self.labels * tf.log(self.output))
                tf.summary.scalar('loss', cross_entropy)
            with tf.name_scope('train'):
                self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            with tf.name_scope('accuracy_train'):
                correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.labels, 1))
                accuracy_train = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy_train', accuracy_train)

    def train(self):
        """
        训练函数
        """
        self.sess = tf.Session()

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('logs/train', self.sess.graph)
        test_writer = tf.summary.FileWriter('logs/test', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        for step in range(2001):
            print("step: "+str(step))
            train_batch = self.mnist.train.next_batch(200)
            test_batch = self.mnist.test.next_batch(200)
            self.sess.run(self.train_step, feed_dict={self.images: train_batch[0], self.labels: train_batch[1]})
            if step % 5 == 0:
                train_record = self.sess.run(merged, feed_dict={self.images: train_batch[0], self.labels: train_batch[1]})
                train_writer.add_summary(train_record, step)

                test_record = self.sess.run(merged, feed_dict={self.images: test_batch[0], self.labels: test_batch[1]})
                test_writer.add_summary(test_record, step)

        saver = tf.train.Saver()
        saver.save(self.sess, "save/session.ckpt")

    def restore_module(self):
        """
        恢复神经网络模型
        """
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, "save/session.ckpt")

    def predict(self, data):
        """
        识别数据
        """
        return self.sess.run(tf.argmax(self.output, 1), feed_dict={self.images: data})[0]


if __name__ == '__main__':
    cnn = Rec()
