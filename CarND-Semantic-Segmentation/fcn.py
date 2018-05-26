from os.path import join
import tensorflow as tf


class FCN:
    def __init__(self, batch_size, epochs, lr, regularization_const, vgg_path, kernel_regularizer_param=1e-3):
        self.batch_size = batch_size
        self.epochs = epochs
        self.regularization_const = regularization_const
        self.lr = lr
        self.vgg_path = vgg_path
        self.kernel_regularizer_param = kernel_regularizer_param

        self.global_step = None

        self.features_placeholder = None
        self.features_lengths_placeholder = None
        self.labels_placeholder = None
        self.prediction = None
        self.prediction_class = None
        self.prediction_softmax = None
        self.train_op = None

        self.loss = None
        self.accuracy = None
        self.keep_prob = None

        self.name = "fcn_{batch}_keepProb{keep_prob}_epochs{epochs}_lr{learning_rate}_regconst{regularization_const}".format(
            batch=self.batch_size,
            keep_prob=self.keep_prob,
            epochs=self.epochs,
            learning_rate=self.lr,
            regularization_const=self.regularization_const
        )
        self.save_directory = join("checkpoint/", self.name)

    def load_model(self, sess):
        print("Will restore model from {path}".format(path=self.saveDirectory))
        training_saver = tf.train.Saver(max_to_keep=None)
        training_saver.restore(sess, self.saveDirectory)

    def save_model(self, sess, epoch):
        print("Will save model at epoch {epoch} to {path}".format(epoch=epoch, path=self.save_directory))
        training_saver = tf.train.Saver(max_to_keep=None)
        training_saver.save(sess, self.save_directory, global_step=epoch)

    def train_model(self, sess, other_operations=list(), keep_prob=.5):
        operations_to_run = [self.train_op] + other_operations

        return sess.run(operations_to_run, feed_dict={
            self.keep_prob: keep_prob
        })

    def load_vgg(self, sess, vgg_path):
        """
        Load Pretrained VGG Model into TensorFlow.
        :param sess: TensorFlow Session
        :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
        :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
        """
        #   Use tf.saved_model.loader.load to load the model and weights
        vgg_tag = 'vgg16'
        vgg_input_tensor_name = 'image_input:0'
        vgg_keep_prob_tensor_name = 'keep_prob:0'
        vgg_layer3_out_tensor_name = 'layer3_out:0'
        vgg_layer4_out_tensor_name = 'layer4_out:0'
        vgg_layer7_out_tensor_name = 'layer7_out:0'
        model = tf.saved_model.loader.load(
            sess,
            [vgg_tag],
            vgg_path
        )

        with tf.name_scope("vgg_base"):
            t1 = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
            t2 = sess.graph.sess.graphget_tensor_by_name(vgg_keep_prob_tensor_name)
            t3 = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
            t4 = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
            t5 = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

            tf.summary.image('image', t1)
            tf.summary.histogram('vgg3', t2)
            tf.summary.histogram('vgg3', t3)
            tf.summary.histogram('vgg4', t4)
            tf.summary.histogram('vgg7', t5)

        return t1, t2, t3, t4, t5

    def conv_1x1(self, input_tensor, num_outputs, kernel_regularizer, activation=None, name="conv_1x1"):
        """
        Perform a 1x1 convolution
        :x: 4-Rank Tensor
        :return: TF Operation
        """
        with tf.name_scope(name):
            kernel_size = 1
            stride = 1

            kernel_initializer = tf.truncated_normal_initializer(stddev=0.01)

            conv_1x1_layer = tf.layers.conv2d(input_tensor, num_outputs,
                                              kernel_size=kernel_size,
                                              strides=stride,
                                              padding='SAME',
                                              dilation_rate=(1, 1),
                                              activation=activation,
                                              kernel_initializer=kernel_initializer,
                                              kernel_regularizer=kernel_regularizer,
                                              name=name)

            tf.summary.histogram(name, conv_1x1_layer)

            return conv_1x1_layer

    def upsample_layer(self, input_tensor, num_outputs, ksize, strides, kernel_regularizer_param, name="upsample"):
        with tf.name_scope(name):
            upsample = tf.layers.conv2d_transpose(input_tensor,
                      num_outputs, ksize, strides,
                      padding="same",
                      kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_regularizer_param),
                      name=name)

            tf.summary.histogram(name, upsample)

            return upsample

    def skip_layer(self, input_tensor1, input_tensor2, name="skip"):
        with tf.name_scope(name):
            skip = tf.add(input_tensor1, input_tensor2)
            tf.summary.histogram(name, skip)
            return skip

    def build_model(self, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        layer7_conv1x1 = self.conv_1x1(vgg_layer7_out, num_classes,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.kernel_regularizer_param),
                                  activation=None, name="layer7_conv1x1")
        upsample_layer7 = self.upsample_layer(layer7_conv1x1, num_classes, 4, 2,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(self.kernel_regularizer_param),
                                 name="upsample_layer7")
        layer4_conv1x1 = self.conv_1x1(vgg_layer4_out, num_classes,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.kernel_regularizer_param),
                                  activation=None, name="layer4_conv1x1")
        layer3_conv1x1 = self.conv_1x1(vgg_layer3_out, num_classes,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(self.kernel_regularizer_param),
                                  activation=None, name="layer3_conv1x1")
        skip1 = self.skip_layer(upsample_layer7, layer4_conv1x1, name="skip1")

        upsample_layer4 = self.upsample_layer(skip1, num_classes, 4, 2,
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(self.kernel_regularizer_param),
                                         name="upsample_layer4")
        skip2 = self.skip_layer(upsample_layer4, layer3_conv1x1, name="skip2")

        last_layer = tf.layers.conv2d_transpose(skip2, num_classes, 16, strides=(8, 8), padding="same",
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(self.kernel_regularizer_param))

        return last_layer

    def optimize(self, nn_last_layer, correct_label, learning_rate, num_classes, reg_const):
        """
        Build the TensorFLow loss and optimizer operations.
        :param nn_last_layer: TF Tensor of the last layer in the neural network
        :param correct_label: TF Placeholder for the correct label image
        :param learning_rate: TF Placeholder for the learning rate
        :param num_classes: Number of classes to classify
        :return: Tuple of (logits, train_op, cross_entropy_loss)
        """
        logits = tf.reshape(nn_last_layer, (-1, num_classes))
        labels = tf.reshape(correct_label, [-1, num_classes])
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        regularisation_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regulatisation_constant = reg_const  # Choose an appropriate one.

        with tf.name_scope('total_loss'):
            loss = cross_entropy_loss + regulatisation_constant * sum(regularisation_losses)
            tf.summary.scalar("loss", loss)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=self.global_step)

        return logits, optimizer, loss



    def train(self):
        with tf.Session() as sess:
