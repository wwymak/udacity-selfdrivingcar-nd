import os
import tensorflow as tf
import helper
from tqdm import tqdm
import warnings
from distutils.version import LooseVersion
# import project_tests as tests


# Check TensorFlow Version
# assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
# print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
# if not tf.test.gpu_device_name():
#     warnings.warn('No GPU found. Please use a GPU to train your neural network.')
# else:
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def load_vgg(sess, vgg_path):
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
        t2 = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        t3 = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
        t4 = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
        t5 = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

        tf.summary.image('image', t1)
        tf.summary.histogram('vgg3', t3)
        tf.summary.histogram('vgg4', t4)
        tf.summary.histogram('vgg7', t5)

    print('vgg loaded',
          t1.get_shape(),
          t2.get_shape(),
          t3.get_shape(),
          t4.get_shape(),
          t5.get_shape())

    return t1, t2, t3, t4, t5


def conv_1x1(input_tensor, num_outputs, kernel_regularizer, activation=None, name="conv_1x1"):
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
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name)

        tf.summary.histogram(name, conv_1x1_layer)

        print(name, ' conv1x1')

        return conv_1x1_layer


def upsample_layer(input_tensor, num_outputs, ksize, strides, kernel_regularizer, name="upsample"):
    with tf.name_scope(name):
        upsample = tf.layers.conv2d_transpose(input_tensor, num_outputs, ksize, strides, padding="same",
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                        name=name)

        tf.summary.histogram(name, upsample)
        print('upsampe ', name )
        return upsample


def skip_layer(input_tensor1, input_tensor2, name="skip"):
    with tf.name_scope(name):
        skip = tf.add(input_tensor1, input_tensor2)
        tf.summary.histogram(name, skip)
        print('skip layer ', name)
        return skip


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    layer7_conv1x1 = conv_1x1(vgg_layer7_out, num_classes, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                              activation=None, name="layer7_conv1x1")
    upsample_layer7 = upsample_layer(layer7_conv1x1, num_classes, 4, 2,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name="upsample_layer7")
    layer4_conv1x1 = conv_1x1(vgg_layer4_out, num_classes, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                              activation=None, name="layer4_conv1x1")
    layer3_conv1x1 = conv_1x1(vgg_layer3_out, num_classes, kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                              activation=None, name="layer3_conv1x1")
    skip1 = skip_layer(upsample_layer7, layer4_conv1x1, name="skip1")

    upsample_layer4 = upsample_layer(skip1, num_classes, 4, 2,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name="upsample_layer4")
    skip2 = skip_layer(upsample_layer4, layer3_conv1x1, name="skip2")
    output_layer = tf.layers.conv2d_transpose(skip2, num_classes, 16, strides=(8, 8), padding="same",
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    print('building layers', output_layer.get_shape().as_list())

    return output_layer

# tests.test_layers(layers)





def optimize(nn_last_layer, correct_label, learning_rate, num_classes, reg_const=1e-3):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, int(num_classes)), name='logits')
    labels = tf.reshape(correct_label, (-1, int(num_classes)), name='ground_truth')
    print(nn_last_layer, correct_label, '1b')
    # logits = tf.reshape(nn_last_layer, [160*576, int(num_classes)])
    # labels = tf.reshape(correct_label, [160*576, int(num_classes)])

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels), name='xent_loss')
    regularisation_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regulatisation_constant = reg_const  # Choose an appropriate one.
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    with tf.name_scope('total_loss'):
        loss = cross_entropy_loss + regulatisation_constant * sum(regularisation_losses)
        tf.summary.scalar("loss", loss)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    print('optimising')
    return logits, optimizer, loss, global_step
# tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, kprob_actual, lr_actual, global_step, saver):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param kprob_actual: value for keep probability
    :param lr_actual: value for learning rate
    """
    # saver = tf.train.Saver()
    tfboard_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('tensorboard_graphs/fcn')
    writer.add_graph(sess.graph)


    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    for epoch in tqdm(range(epochs)):
        step = 0
        for imgs, gt_labels in get_batches_fn(batch_size):
            print(imgs.shape, gt_labels.shape)
            _, total_loss, summary = sess.run([train_op, cross_entropy_loss, tfboard_summary], feed_dict={

                learning_rate: 0.01,
                correct_label: gt_labels,
                keep_prob: 0.5,
                input_image: imgs})
            writer.add_summary(summary, global_step=epoch)
            print("loss: ", total_loss, " step: ", step, " epoch: ", epoch)
            step += 1
    saver.save(sess, 'checkpoints/fcn', global_step=epoch)
    # return saver


# tests.test_train_nn(train_nn)

# def test():
#     num_classes = 2
#     image_shape = (160, 576)
#     data_dir = './data'
#     runs_dir = './runs'
#     tests.test_load_vgg(load_vgg, tf)
#     tests.test_layers(layers)
#     tests.test_optimize(optimize)
#     tests.test_train_nn(train_nn)
#     tests.test_for_kitti_dataset(data_dir)


def run():
    num_classes = 2
    image_shape = (160, 160)
    # image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    safe_mkdir('checkpoints')
    safe_mkdir('tensorboard_graphs')



    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/



    graph = tf.Graph()
    with graph.as_default():


        with tf.Session(graph=graph) as sess:
        # with tf.Session() as sess:
            learning_rate = tf.placeholder(tf.float32)
            correct_label = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], num_classes))
            # input_image = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], 3))
            reg_const = 0.1
            epochs = 5
            batch_size = 4
            kprob_actual = 0.5
            lr_actual = 0.01

            # Path to vgg model
            vgg_path = os.path.join(data_dir, 'vgg')
            # Create function to get batches
            get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

            # OPTIONAL: Augment Images for better results
            #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

            # if a checkpoint exists, restore from the latest checkpoint

            image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
            nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

            logits, optimizer, loss, global_step = optimize(nn_last_layer, correct_label, learning_rate, reg_const)

            saver = tf.train.Saver()
            print('1')
            train_nn(sess, epochs, batch_size, get_batches_fn, optimizer, loss, image_input,
                     correct_label, keep_prob, learning_rate, kprob_actual, lr_actual, global_step, saver)
            print('2')
            # TODO: Save inference data using helper.save_inference_samples
            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    # test()
    run()


