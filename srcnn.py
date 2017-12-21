from preprocess import *
import os
import time
import tensorflow as tf

try:
    xrange
except:
    xrange = range


class srcnn:
    def __init__(self,
                 sess,
                 input_image_size=33,
                 ground_truth_size=21,
                 batch_size=128,
                 channel_dim=1,
                 checkpoint_dir=None,
                 samples_dir=None):

        self.sess = sess
        self.is_grayscale = (channel_dim == 1)
        self.input_image_size = input_image_size
        self.ground_truth_size = ground_truth_size
        self.batch_size = batch_size
        self.channel_dim = channel_dim
        self.checkpoint_dir = checkpoint_dir
        self.samples_dir = samples_dir

        self.input_images = None
        self.ground_truths = None
        self.weights = None
        self.biases = None
        self.model = None
        self.loss = None
        self.training_optimizer = None
        self.saver = None

        self.setup()

    def setup(self):
        self.input_images = tf.placeholder(tf.float32, [None, self.input_image_size, self.input_image_size,
                                                        self.channel_dim], name='input_images')
        self.ground_truths = tf.placeholder(tf.float32, [None, self.ground_truth_size, self.ground_truth_size,
                                                         self.channel_dim], name='ground_truths')

        self.weights = {
            'W1': tf.Variable(tf.truncated_normal(shape=[9, 9, 1, 64], mean=0, stddev=1e-3, name='W1')),
            'W2': tf.Variable(tf.truncated_normal(shape=[1, 1, 64, 32], mean=0, stddev=1e-3, name='W2')),
            'W3': tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 1], mean=0, stddev=1e-3, name='W3'))
        }

        self.biases = {
            'b1': tf.Variable(tf.zeros([64]), name='b1'),
            'b2': tf.Variable(tf.zeros([32]), name='b2'),
            'b3': tf.Variable(tf.zeros([1]), name='b3')
        }

        conv_1 = tf.nn.relu(tf.nn.conv2d(self.input_images, self.weights['W1'], strides=[1, 1, 1, 1],
                                         padding='VALID') + self.biases['b1'])
        conv_2 = tf.nn.relu(tf.nn.conv2d(conv_1, self.weights['W2'], strides=[1, 1, 1, 1],
                                         padding='VALID') + self.biases['b2'])
        conv_3 = tf.nn.conv2d(conv_2, self.weights['W3'], strides=[1, 1, 1, 1],
                              padding='VALID') + self.biases['b3']

        self.model = conv_3

        # MSE Loss
        self.loss = tf.reduce_mean(tf.square(self.ground_truths - self.model))

        self.saver = tf.train.Saver()

    def train(self, config):
        if config.is_train:
            input_setup(self.sess, config)
        else:
            nx, ny = input_setup(self.sess, config)

        if config.is_train:
            data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
        else:
            data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")

        train_data, train_ground_truth = read_data(data_dir)

        self.training_optimizer = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)

        tf.global_variables_initializer().run()

        num_iteration = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print("Load successfully!")
        else:
            print("!!! NO FILES FOR LOADING !!!")

        if config.is_train:
            print("TRAINING ... ...")

            for epoch in xrange(config.epoch):
                batch_indices = len(train_data) // config.batch_size
                for index in xrange(0, batch_indices):
                    start = index * config.batch_size
                    end = (index+1) * config.batch_size
                    batch_images = train_data[start: end]
                    batch_ground_truths = train_ground_truth[start: end]

                    num_iteration += 1

                    _, loss_value = self.sess.run([self.training_optimizer, self.loss],
                                                  feed_dict={self.input_images: batch_images,
                                                             self.ground_truths: batch_ground_truths})

                    if num_iteration % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]"
                              % (epoch+1, num_iteration, time.time() - start_time, loss_value))

                    if num_iteration % 500 == 0:
                        self.save(config.checkpoint_dir, num_iteration)

        else:
            print("TESTING ... ...")
            result = self.model.eval({self.input_images: train_data, self.ground_truths: train_ground_truth})

            result = merge(result, [nx, ny])
            result = result.squeeze()
            image_path = os.path.join(os.getcwd(), config.samples_dir)
            image_path = os.path.join(image_path, "test_image.png")
            imsave(result, image_path)

    def load(self, checkpoint_dir):
        print("Reading checkpoints ...")
        model_dir = "%s_%s" % ("srcnn", self.ground_truth_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def save(self, checkpoint_dir, step):
        model_name = "srcnn.model"
        model_dir = "%s_%s" % ("srcnn", self.ground_truth_size)

        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
