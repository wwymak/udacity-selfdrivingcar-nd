import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# TODO for whatever reason this keras implementation trains really slowy and have memory issues-- need to debug properly

import tensorflow as tf
# from tensorboard
from keras.models import *
from keras.layers import *
from keras.applications.vgg16 import VGG16
from keras.regularizers import l2

from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
from keras.optimizers import Adam, SGD
from keras.callbacks import *
from keras_contrib.layers import crf

import glob
import numpy as np
import os
from PIL import Image
from skimage.transform import resize
from imageio import imread

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps     # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros((self.nb_steps * ib.shape[0], *ib.shape[1:]), dtype=np.float32)
                tags = np.zeros((self.nb_steps * tb.shape[0], *tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super().on_epoch_end(epoch, logs)


class FCN:
    def __init__(self, num_classes=2, input_shape=(160, 576, 3), batch_size=4, epochs=2, lr=0.001,
                 checkpoint_dir='checkpoints',weight_decay=5e-4,initial_epoch=0,
                 model = None, train_img_files_list=None, train_gt_list=None,
                 val_img_files_list=None, val_gt_list=None):
        self.num_classes = num_classes
        self.input_shape=input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.initial_epoch = initial_epoch
        self.weight_decay= weight_decay
        self.checkpoint_dir = checkpoint_dir
        self.model = model
        self.train_img_files_list = train_img_files_list
        self.train_gt_list = train_gt_list
        self.val_img_files_list = val_img_files_list
        self.val_gt_list = val_gt_list
        assert train_img_files_list is not None, "specify training img files"
        assert train_gt_list is not None, "specify training ground truth files"
        assert val_img_files_list is not None, "specify validation img files"
        assert val_gt_list is not None, "specify validation ground truth files"

    def train_generator(self, files_list, gt_list, batch_size, image_shape):

        background_color = np.array([1, 0, 0])
        images = []
        gt_images = []

        while True:
            for i in range(batch_size):
                index = np.random.choice(len(files_list), replace=False)
                image_file = files_list[index]
                gt_file = gt_list[index]
                gt_image = resize(imread(gt_file), image_shape)

                image = resize(imread(image_file), image_shape)

                gt_bg_mask = np.all(gt_image == background_color, axis=2)
                gt_image_bg = np.zeros(shape=(image_shape[0], image_shape[1]))
                gt_image_fg = np.ones(shape=(image_shape[0], image_shape[1]))

                gt_image_fg[gt_bg_mask] = 0
                gt_image_bg[gt_bg_mask] = 1

                gt_onehot = np.stack((gt_image_fg, gt_image_bg), axis=-1)

                images.append(image)
                gt_images.append(gt_onehot)

            yield np.array(images), np.array(gt_images)

    def fcn_vgg_8(self, input_shape, num_classes, weight_decay=5e-4):
        img_input = Input(shape=input_shape)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        vgg_conv3_out = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(vgg_conv3_out)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        vgg_conv4_out = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(vgg_conv4_out)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        vgg_conv5_out = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        # convert dense layers to fully conv layers
        x = Conv2D(4096, (7, 7), dilation_rate=(2, 2), activation='relu', padding='same', name='fc1',
                   kernel_regularizer=l2(weight_decay))(vgg_conv5_out)
        x = Dropout(0.5)(x)
        x = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fc2', kernel_regularizer=l2(weight_decay))(x)
        x = Dropout(0.5)(x)
        conv7_out = Conv2D(num_classes, (1, 1), activation='linear', padding='valid', strides=(1, 1),
                           kernel_regularizer=l2(weight_decay), name='decoder_conv7')(x)

        # decoder network
        vgg_conv4_conv1x1 = Conv2D(num_classes, (1, 1), activation='relu', padding='valid', strides=(1, 1),
                                   kernel_regularizer=l2(weight_decay),
                                   name="layer4_conv1x1")(vgg_conv4_out)

        vgg_conv3_conv1x1 = Conv2D(num_classes, (1, 1), activation='relu', padding='valid', strides=(1, 1),
                                   kernel_regularizer=l2(weight_decay),
                                   name="layer3_conv1x1")(vgg_conv3_out)

        upsample1 = Conv2DTranspose(num_classes, kernel_size=4, strides=(2, 2), padding='same',
                                    kernel_regularizer=l2(weight_decay), name='upsample_1')(conv7_out)

        skip1 = Add()([upsample1, vgg_conv4_conv1x1])
        upsample2 = Conv2DTranspose(num_classes, kernel_size=4, strides=(2, 2), padding='same',
                                    kernel_regularizer=l2(weight_decay), name='upsample_2')(skip1)

        skip2 = Add()([upsample2, vgg_conv3_conv1x1])
        upsample3 = Conv2DTranspose(num_classes, kernel_size=16, strides=(8, 8), padding='same',
                                    kernel_regularizer=l2(weight_decay), name='upsample_3')(skip2)

        model = Model(img_input, upsample3)

        layers_list = model.layers
        index = {}

        for layer in layers_list:
            if layer.name:
                index[layer.name] = layer

        vgg16 = VGG16()
        for layer in vgg16.layers:
            weights = layer.get_weights()
            if layer.name == 'fc1':
                weights[0] = np.reshape(weights[0], (7, 7, 512, 4096))
            elif layer.name == 'fc2':
                weights[0] = np.reshape(weights[0], (1, 1, 4096, 4096))
            if layer.name in index:
                index[layer.name].set_weights(weights)
                index[layer.name].trainable = False

        return model

    def binary_crossentropy_loss(self, ground_truth, predictions):
        return K.mean(K.binary_crossentropy(ground_truth,
                                            predictions,
                                            from_logits=True), axis=-1)

    def loss_with_regularisation(self, ground_truth, predictions):
        cross_entropy_loss = self.binary_crossentropy_loss(ground_truth, predictions)
        regularisation_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regulatisation_constant = self.weight_decay  # Choose an appropriate one.
        return cross_entropy_loss + regulatisation_constant * sum(regularisation_losses)

    def mean_IoU(self, y_true, y_pred):
        s = K.shape(y_true)
        y_true_reshaped = K.reshape(y_true, tf.stack([-1, s[1] * s[2], s[-1]]))
        y_pred_reshaped = K.reshape(y_pred, tf.stack([-1, s[1] * s[2], s[-1]]))
        y_pred_argmax = K.argmax(y_pred_reshaped, axis=-1)
        clf_pred = K.one_hot(y_pred_argmax, s[-1])

        equal_entries = K.cast(K.equal(clf_pred, y_true_reshaped), dtype='float32') * y_true_reshaped

        intersection = K.sum(equal_entries, axis=1)
        union_per_class = K.sum(y_true_reshaped, axis=1) + K.sum(clf_pred, axis=1)

        iou = intersection / (union_per_class - intersection)
        iou_mask = tf.is_finite(iou)
        iou_masked = tf.boolean_mask(iou, iou_mask)

        return K.mean(iou_masked)

    def create_or_load_model(self):
        if self.model is not None:
            print('load model', self.model)
            model_fcn = load_model(self.model,
                    custom_objects={'binary_crossentropy_loss': self.binary_crossentropy_loss,
                                    'loss_with_regularisation': self.loss_with_regularisation,
                                    'mean_IoU': self.mean_IoU})
        else:
            model_fcn = self.fcn_vgg_8(self.input_shape, self.num_classes, weight_decay=5e-4)
            # freeze layers from vgg for now
            for idx, layer in enumerate(model_fcn.layers):
                if idx < 17:
                    layer.trainable = False
            # model_fcn.summary()

            model_fcn.compile(loss=self.loss_with_regularisation,
                              # loss=self.binary_crossentropy_loss,
                              optimizer=Adam(lr=self.lr),
                              metrics=[self.mean_IoU]
                              )
        return model_fcn

    def train_model(self):
        safe_mkdir(self.checkpoint_dir)
        tensorboardCB = TensorBoard(log_dir=os.path.join('tensorboard_graphs', 'fcn-keras-lr{}'.format(self.lr)), write_graph=True)
        checkpointCB = ModelCheckpoint(self.checkpoint_dir + '/fcn-run5-lr{}.h5'.format(self.lr), save_best_only=True,
                                       monitor='mean_IoU', mode='max')

        callbacks = [tensorboardCB, checkpointCB]
        t_generator = self.train_generator(self.train_img_files_list, self.train_gt_list, self.batch_size,
                                           self.input_shape)
        val_generator = self.train_generator(self.val_img_files_list, self.val_gt_list,
                                             self.batch_size, self.input_shape)

        # for i in range(self.epochs):
        #     K.clear_session()
        #     model_fcn = self.create_or_load_model()
        #     print('model loaded')
        #     validation_data = val_generator.__next__()
        #     history = model_fcn.fit_generator(t_generator, epochs=i+1,
        #                                       # steps_per_epoch=4,
        #                                       steps_per_epoch=len(self.train_img_files_list)//self.batch_size,
        #                                       validation_data=validation_data,
        #                                       # validation_data=val_generator, validation_steps=2,
        #                                       callbacks=callbacks,
        #                                       initial_epoch=i)
        #
        #     self.model = self.checkpoint_dir + '/fcn-run5.h5' #.format(i+1)
        model_fcn = self.create_or_load_model()
        print('model loaded')
        validation_data = val_generator.__next__()
        history = model_fcn.fit_generator(t_generator, epochs=self.epochs,
                                          # steps_per_epoch=4,
                                          steps_per_epoch=len(self.train_img_files_list) // self.batch_size,
                                          validation_data=validation_data,
                                          # validation_data=val_generator, validation_steps=2,
                                          callbacks=callbacks,
                                          initial_epoch=0)




if __name__ == '__main__':
    os.environ["TF_CUDNN_WORKSPACE_LIMIT_IN_MB"] = '100'
    np.random.seed(10)
    road_imgs_gt = glob.glob(os.path.join('./data', 'data_road/training', 'gt_image_2', '*_road_*.png'))
    np.random.shuffle(road_imgs_gt)
    val_gt_arr = road_imgs_gt[:16]
    val_img_arr = [x.replace('gt_image_2', 'image_2').replace('_road_', '_').replace('_lane_', '_') for x in val_gt_arr]
    train_gt_arr = road_imgs_gt[16:]
    train_img_arr = [x.replace('gt_image_2', 'image_2').replace('_road_', '_').replace('_lane_', '_') for x in
                     train_gt_arr]

    fcn = FCN(num_classes=2, input_shape=(64, 256, 3), batch_size=2, epochs=10, lr=0.01,
                train_img_files_list=train_img_arr, train_gt_list=train_gt_arr, model='checkpoints/fcn-run5-lr0.01.h5',
              val_img_files_list=val_img_arr,
              val_gt_list=val_gt_arr, initial_epoch=3)

    fcn.train_model()



