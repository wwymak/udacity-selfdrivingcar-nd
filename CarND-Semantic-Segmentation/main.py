import tensorflow as tf
import helper
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from distutils.version import LooseVersion
import project_tests as tests
from tqdm import tqdm

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))



l2_weight_decay = 1e-5
reg_const = 1e-5
# learning_rate_val = 0.01
learning_rate_val = 0.001
keep_prob_val = 0.5
num_epochs = 40
batch_size = 8
num_classes = 2
image_shape = (160, 576)

data_dir = './data'
runs_dir = './runs'
train_dir = './data/data_road/training'
vgg_path = './data/vgg'

freeze_vgg_top = True

tf_graph_dir = 'tensorboard_graphs/fcn_lr-{}_kp-{}_batch-{}_regconst-{}_vggfrozen-{}'.format(learning_rate_val,
                                                                                             keep_prob_val,
                                                                                             batch_size, reg_const,
                                                                                             freeze_vgg_top)
checkpoint_dir = 'checkpoints/fcn_lr-{}_kp-{}_batch-{}_regconst-{}_vggfrozen-{}'.format(learning_rate_val,
                                                                                        keep_prob_val,
                                                                                        batch_size, reg_const,
                                                                                        freeze_vgg_top)


def load_vgg(sess, vgg_path, freeze_vgg_top=False):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
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

    graph = tf.get_default_graph()
    with tf.name_scope("vgg_base"):
        image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
        keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
        layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
        layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
        layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    if freeze_vgg_top:
        layer7 = tf.stop_gradient(layer7)
        layer4 = tf.stop_gradient(layer4)
        layer3 = tf.stop_gradient(layer3)

    tf.summary.image('image', image_input)

    return image_input, keep_prob, layer3, layer4, layer7


# test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    with tf.name_scope("decoder"):
        layer7_conv1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes,
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_weight_decay),
                                          kernel_size=1,
                                          strides=1,
                                          # padding='SAME',
                                          name="layer7_conv1x1")

        layer8_upsample = tf.layers.conv2d_transpose(layer7_conv1x1, vgg_layer4_out.get_shape().as_list()[-1],
                                                     kernel_size=4,
                                                     strides=2,
                                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                         l2_weight_decay),
                                                     padding='SAME',
                                                     name="layer8_upsample")

        layer9_skip = tf.add(layer8_upsample, vgg_layer4_out, name="layer9_skip")

        layer10_upsample = tf.layers.conv2d_transpose(layer9_skip, vgg_layer3_out.get_shape().as_list()[-1],
                                                      kernel_size=4,
                                                      strides=2,
                                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                                          l2_weight_decay),
                                                      padding='SAME',
                                                      name="layer10_upsample")
        layer11_skip = tf.add(layer10_upsample, vgg_layer3_out, name="layer11_skip")

        layer12_conv1x1 = tf.layers.conv2d_transpose(layer11_skip, num_classes,
                                                     kernel_size=16,
                                                     strides=8,
                                                     padding='SAME',
                                                     name="layer12_conv1x1")
        tf.summary.histogram('layer7_conv1x1', layer7_conv1x1)

        tf.summary.histogram('layer7_conv1x1', layer7_conv1x1)
        tf.summary.histogram('layer8_upsample', layer8_upsample)
        tf.summary.histogram('layer9_skip', layer9_skip)
        tf.summary.histogram('layer10_upsample', layer10_upsample)
        tf.summary.histogram('layer11_skip', layer11_skip)
        tf.summary.histogram('layer12_conv1x1', layer12_conv1x1)

        return layer12_conv1x1


# test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    labels = tf.reshape(correct_label, (-1, num_classes), name='ground_truth')
    probabilities = tf.nn.softmax(logits, name='logits_to_prediction')
    predictions = tf.argmax(probabilities, -1)
    print(probabilities.get_shape().as_list(), labels.get_shape().as_list(), predictions.get_shape().as_list())
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),
                                        name='xent_loss')
    regularisation_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regulatisation_constant = reg_const

    cross_entropy_loss = cross_entropy_loss + regulatisation_constant * sum(regularisation_losses)
    mean_IOU, mean_IOU_update = tf.contrib.metrics.streaming_mean_iou(predictions=predictions,
                                                                      labels=tf.argmax(labels, -1),
                                                                      num_classes=num_classes)

    tf.summary.scalar("loss", cross_entropy_loss)
    tf.summary.scalar("mean_iou", mean_IOU)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss, name='train_op')

    return logits, optimizer, cross_entropy_loss, mean_IOU_update


# test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, iou_op):
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
    """

    # saver for stashing checkpoints
    saver = tf.train.Saver(allow_empty=True)

    tfboard_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tf_graph_dir, graph=sess.graph)
    writer.add_graph(sess.graph)

    #     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #     if ckpt and ckpt.model_checkpoint_path:
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    step = 0
    for epoch in tqdm(range(epochs)):

        total_loss = 0
        for imgs, gt_labels in get_batches_fn(batch_size):
            # only run the summary op every 10 steps to reduce computation
            if step % 10:
                _, loss, iou, summary = sess.run([train_op, cross_entropy_loss, iou_op, tfboard_summary], feed_dict={
                    input_image: imgs,
                    correct_label: gt_labels,
                    keep_prob: keep_prob_val,
                    learning_rate: learning_rate_val})
                writer.add_summary(summary, global_step=step)
            else:
                _, loss, iou = sess.run([train_op, cross_entropy_loss, iou_op], feed_dict={
                    input_image: imgs,
                    correct_label: gt_labels,
                    keep_prob: keep_prob_val,
                    learning_rate: learning_rate_val})

            step += 1
            total_loss += loss
        print("loss: ", loss, " step: ", step, " epoch: ", epoch)
        # save a checkpoint per epoch
        saver.save(sess, checkpoint_dir, global_step=step)


# test_train_nn(train_nn)


def run():
    #     test_for_kitti_dataset(data_dir)

    helper.safe_mkdir('checkpoints')
    helper.safe_mkdir('tensorboard_graphs')
    helper.clear_gpu()
    # clear out the graph from previous runs
    tf.reset_default_graph()

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)
    # define data generator
    get_batches_fn = helper.gen_batch_function(train_dir, image_shape)

    with tf.Session() as sess:
        # input placeholders
        learning_rate = tf.placeholder(tf.float32)
        correct_label = tf.placeholder(tf.float32, shape=(None, image_shape[0], image_shape[1], num_classes))
        keep_prob = tf.placeholder(tf.float32)

        # loading the pretrained vgg16 backbone
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path, freeze_vgg_top)
        # decoder construction
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        # get the training op and the output
        logits, optimizer, loss, iou_op = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # initialise variabes
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Train fcn network
        train_nn(sess, num_epochs, batch_size, get_batches_fn,
                 optimizer, loss, image_input,
                 correct_label, keep_prob, learning_rate, iou_op)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

        print('done')


if __name__ == '__main__':
    run()