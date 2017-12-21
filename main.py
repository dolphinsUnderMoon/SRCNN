from srcnn import *
from preprocess import *
import tensorflow as tf
import os
import pprint

# set the hyper parameters
flags = tf.app.flags
flags.DEFINE_integer("epoch", 10000, "Number of epochs [10000]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("input_image_size", 33, "The size of input image to use [33]")
flags.DEFINE_integer("ground_truth_size", 21, "The size of image (high resolution) to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("channel_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("samples_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


def main(_):
    # print the setting of the model
    pp.pprint(flags.FLAGS.__flags)

    # create corresponding checkpoint and samples directory
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.samples_dir):
        os.makedirs(FLAGS.samples_dir)

    with tf.Session() as sess:
        srcnn_model = srcnn(sess,
                            input_image_size=FLAGS.input_image_size,
                            ground_truth_size=FLAGS.ground_truth_size,
                            batch_size=FLAGS.batch_size,
                            channel_dim=FLAGS.channel_dim,
                            checkpoint_dir=FLAGS.checkpoint_dir,
                            samples_dir=FLAGS.samples_dir)

        srcnn_model.train(FLAGS)


if __name__ == "__main__":
    tf.app.run()

