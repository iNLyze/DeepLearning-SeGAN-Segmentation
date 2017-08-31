
# This work was inspired by Xue et al. http://arxiv.org/abs/1706.01805 as well as the excellent "Deep Learning for coders" tought by Jeremy Howard and Rachel Thomas on http://course.fast.ai/

from keras.engine import Model
from keras.layers import Lambda
from keras.layers import Dropout, LeakyReLU, Input, Activation, BatchNormalization, Concatenate, multiply, Flatten
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.regularizers import l1, l2
from keras.losses import mae
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook



def dropout(x, p):
    return Dropout(p)(x)


def bnorm(x):
    return BatchNormalization()(x)


def relu(x):
    return Activation('relu')(x)


def conv_l1(x, nb_filters, kernel, stride=(1, 1)):
    return Convolution2D(nb_filters, kernel, padding='same',
                         kernel_initializer='he_uniform',
                         kernel_regularizer=l1(0.01), strides=(stride, stride))(x)


def convl1_lrelu(x, nb_filters, kernel, stride):
    x = conv_l1(x, nb_filters, kernel, stride)
    return LeakyReLU()(x)


def convl1_bn_lrelu(x, nb_filters, kernel, stride):
    x = conv_l1(x, nb_filters, kernel, stride)
    x = bnorm(x)
    return LeakyReLU()(x)


def shared_convl1_lrelu(shape, nb_filters, kernel, stride=(1, 1), **kwargs):
    # i = Input(shape)
    c = Convolution2D(nb_filters, kernel, padding='same',
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l1(0.01), strides=(stride, stride), input_shape=shape)
    l = LeakyReLU()
    return Sequential([c, l], **kwargs)


def shared_convl1_bn_lrelu(shape, nb_filters, kernel, stride=(1, 1), **kwargs):
    c = Convolution2D(nb_filters, kernel, padding='same',
                      kernel_initializer='he_uniform',
                      kernel_regularizer=l1(0.01), strides=(stride, stride), input_shape=shape)
    b = BatchNormalization()
    l = LeakyReLU()
    return Sequential([c, b, l], **kwargs)


def upsampl_block(x, nb_filters, kernel, stride, size):
    x = UpSampling2D(size=size)(x)
    x = conv_l1(x, nb_filters, kernel, stride)
    x = bnorm(x)
    return relu(x)


def upsampl_conv(x, nb_filters, kernel, stride, size):
    x = UpSampling2D(size=size)(x)
    return conv_l1(x, nb_filters, kernel, stride)

def upsampl_softmax(x, nb_filters, kernel, stride, size, max_project=True):
    x = UpSampling2D(size=size)(x)
    x = conv_l1(x, nb_filters, kernel, stride)
    x = Lambda(hidim_softmax, name='softmax')(x)
    if max_project:
        x = Lambda(lambda x: K.max(x, axis=-1, keepdims=True), name='MaxProject')(x)
    return x


def level_block(previous_block, nb_filters, depth, filter_inc_rate, p):
    print('Current level block depth {}'.format(depth))
    # curr_block = previous_block
    curr_block = convl1_bn_lrelu(previous_block, nb_filters, 4, 2)
    print('Shape prev {}, shape curr {} and depth {} before recursion'.format(previous_block.shape, curr_block.shape,
                                                                              depth))
    curr_block = dropout(curr_block, p) if p else curr_block
    if depth > 0:  # Call next recursion level
        curr_block = level_block(curr_block, int(filter_inc_rate * nb_filters), depth - 1, filter_inc_rate, p)
    print('Shape prev {}, shape curr {} and depth {} after recursion'.format(previous_block.shape, curr_block.shape,
                                                                             depth))
    curr_block = upsampl_block(curr_block, nb_filters, 3, 1, 2)
    print('Shape prev {}, shape curr {} and depth {} after upsampling'.format(previous_block.shape, curr_block.shape,
                                                                              depth))
    curr_block = Concatenate(axis=3)([curr_block, previous_block])
    print('Shape curr {} and depth {} before return'.format(curr_block.shape, depth))
    return curr_block

def hidim_softmax(x, axis=-1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply hidim_softmax to a tensor that is 1D')

class SeGAN(object):
    def __init__(self, imgs, gt, start_filters=64, filter_inc_rate=2, out_ch=1, depth=2,
                 optimizer=RMSprop(2e-5), loss=mae, softmax=False, crop=True, max_project=True):
        """

        :type imgs: Collection of images, numpy array 
        """
        # Note: Future improvement is to allow bcolz arrays as well
        self.imgs = imgs  # Images
        self.gt = gt  # images cropped with ground-truth mask
        self.n = imgs.shape[0]  # number of images
        self.shape = imgs.shape[1:]  # Shape of one image
        self.optimizer = optimizer
        self.loss = loss
        self.softmax = softmax
        self.crop = crop
        self.max_project = max_project
        self.netS = self.segmentor(start_filters, filter_inc_rate, out_ch, depth)
        self.netC = self.critic()
        if max_project:
            self.model = Sequential([self.netS, self.netC], name='segan_model')
        self.dl, self.gl, self.rl, self.fl = [], [], [], []
        self.gen_iterations = 0  # Counter for generator training iterations


    def segmentor(self, start_filters=64, filter_inc_rate=2, out_ch=1, depth=2):
        """
        Creates recursively a segmentor model a.k.a. generator in GAN literature
        """
        inp = Input(shape=self.shape)
        first_block = convl1_lrelu(inp, start_filters, 4, 2)
        middle_blocks = level_block(first_block, int(start_filters * 2), depth=depth,
                                    filter_inc_rate=filter_inc_rate, p=0.1)
        if self.softmax:
            last_block = upsampl_softmax(middle_blocks, out_ch+1, 3, 1, 2, self.max_project) # out_ch+1, because softmax needs crossentropy
        else:
            last_block = upsampl_conv(middle_blocks, out_ch, 3, 1, 2)
        if self.crop:
            out = multiply([inp, last_block])  # crop input with predicted mask
            return Model([inp], [out], name='segmentor_net')
        return Model([inp], [last_block], name='segmentor_net')
        #return Model([inp], [last_block], name='segmentor_net')

    def critic(self):
        """
        Creates a critic a.k.a. discriminator model
        """
        # Note: Future improvement is to provide definable depth of critic
        inp_cropped = Input(self.shape, name='inp_cropped_image')  # Data cropped with generated OR g.t. mask


        shared_1 = shared_convl1_lrelu(self.shape, 64, 4, 2, name='shared_1_conv_lrelu')
        shared_2 = shared_convl1_bn_lrelu((16, 16, 64), 128, 4, 2, name='shared_2_conv_bn_lrelu')
        shared_3 = shared_convl1_bn_lrelu((8, 8, 128), 256, 4, 2, name='shared_3_conv_bn_lrelu')
        shared_4 = shared_convl1_bn_lrelu((4, 4, 256), 512, 4, 2, name='shared_4_conv_bn_lrelu')

        x1_S = shared_1(inp_cropped)
        #x1_S = shared_1(multiply([inp, mask]))
        x2_S = shared_2(x1_S)
        x3_S = shared_3(x2_S)
        x4_S = shared_4(x3_S)
        features = Concatenate(name='features_S')(
            [Flatten()(inp_cropped), Flatten()(x1_S), Flatten()(x2_S), Flatten()(x3_S), Flatten()(x4_S)]
            #[Flatten()(inp), Flatten()(x1_S), Flatten()(x2_S), Flatten()(x3_S), Flatten()(x4_S)]
        )
        return Model(inp_cropped, features, name='critic_net')
        #return Model([inp, mask], features, name='critic_net')

    def rand_idx(self, n, nb_samples):
        return np.random.randint(0, n, size=nb_samples)

    def ground_truth_mask_to_C(self, nb_samples):
        idx = self.rand_idx(self.n, nb_samples)
        X = self.gt[idx]#*self.imgs[idx] #Cropping
        while len(X.shape)<4: X = np.expand_dims(X, axis=-1)
        X_pred = self.netS.predict(X)
        # Here we feed an image cropped with a predicted mask, needed to compute mae
        # between features of gt and features of S
        #y = self.netC.predict(X_pred)
        y = self.model.layers[-1].predict(X_pred)
        return X, -y  # We return a gt cropped image multi-level features of predicted mask
        # Negative sign, because critic needs to maximize loss

    def uncropped_img_to_C(self, nb_samples):
        # Like ground_truth_mask_to_C, but with non-cropped images
        idx = self.rand_idx(self.n, nb_samples)
        X = self.imgs[idx]
        while len(X.shape) < 4: X = np.expand_dims(X, axis=-1)
        X_pred = self.netS.predict(X)
        # y = self.netC.predict(X_pred)
        y = self.model.layers[-1].predict(X_pred)
        return X, -y

    def mixed_mask_to_C(self, nb_samples, gt_share=0.3):
        """
        Experimental method for feeding labeled and unlabeled data during training,
        i.e. semi-supervised learning
        
        Parameters
        -----------
        
        gt_share: relative proportion of ground truth data for learning
        
        """
        # Feeds gt_share gt_images + (1-gt_share) uncropped images to C
        gt_samples = np.int(np.round(nb_samples*gt_share))
        non_gt_samples = np.int(np.round(nb_samples*(1-gt_share)))
        X_gt, y_gt = self.ground_truth_mask_to_C(gt_samples)
        if non_gt_samples:
            X_non, y_non = self.uncropped_img_to_C(non_gt_samples)
            X = np.concatenate((X_gt, X_non))
            y = np.concatenate((y_gt, y_non))
        else:
            X = X_gt
            y = y_gt
        randomize = np.arange(len(X))
        np.random.shuffle(randomize)
        X = X[randomize]
        y = y[randomize]
        return X, y


    def predicted_mask_to_C(self, nb_samples):
        idx = self.rand_idx(self.n, nb_samples)
        while len(X.shape) < 4: X = np.expand_dims(X, axis=-1)
        X = self.netS.predict(self.imgs[idx])
        # y = self.netC.predict(self.gt[idx]) # Here we feed an image cropped with a g.t. mask
        y = self.model.layers[-1].predict(self.gt[idx])
        return X, y  # Return an image cropped with pred masks and multi-level features of gt mask

    def predicted_mask_to_full(self, nb_samples):
        idx = self.rand_idx(self.n, nb_samples)
        X = self.imgs[idx]
        while len(X.shape) < 4: X = np.expand_dims(X, axis=-1)
        # pred = netS.predict(X)
        # y = self.netC.predict(self.gt[idx]) # Here we feed an image cropped with a g.t. mask
        y = self.model.layers[-1].predict(self.gt[idx])
        return X, y

    def plot_losses(self):
        fig, ax = plt.subplots(nrows=1, ncols=4)
        #fig = plt.figure(figsize=(14.,14.))
        styles = ['bo', 'g+', 'r-', 'y.']
        loss_tuple = (self.dl, self.gl, self.rl, self.fl)
        if not loss_tuple is tuple:  # if not tuple make it a tuple
            loss_tuple = (loss_tuple,)
        for s, l in enumerate(loss_tuple):
            print(ax.shape)
            ax[s].plot(l, styles[s])
        #ax.legend
        plt.show()

    def make_trainable(self, net, val):
        """
        Sets layers in keras model trainable or not
        
        val: trainable True or False
        """
        net.trainable = val
        for l in net.layers: l.trainable = val

    def train(self, nb_epoch=5000, bs=128, first=True, gt_share=0.3, trainC_iters = 5):
        # dl..average loss on discriminator
        # gl..loss on generator
        # rl..discriminator loss on real data
        # fl..discriminator loss on fake data
        # gt_share..proportion of ground truth images for training. Fully supervised if gt_share=1.0
        # trainC_iters..how many iterations the critic gets trained for per epoch of segmentor training


        for e in tqdm_notebook(range(nb_epoch)):
            i = 0  # batch counter
            while i < bs:
                #self.make_trainable(self.netC, True)
                self.make_trainable(self.model.layers[-1], True) # make critic trainable
                d_iters = (100 if first and (self.gen_iterations < 25) or self.gen_iterations % 500 == 0
                           else trainC_iters)
                j = 0
                while j < d_iters and i < bs:
                    # print('d_iter {}, epoch {}'.format(j,i))
                    j += 1
                    i += 1


                    X, y = self.mixed_mask_to_C(bs, gt_share)

                    #self.rl.append(self.netC.train_on_batch(X, y))
                    self.rl.append(self.model.layers[-1].train_on_batch(X, y)) # Train critic

                    # fl.append(D.train_on_batch(X,y))
                    # dl.append(np.mean([rl[-1], fl[-1]])) # average latest value
                    self.dl.append(self.rl[-1])
                    self.fl.append(0)

                #self.make_trainable(self.netC, False)
                self.make_trainable(self.model.layers[-1], False) # Fix weights on critic
                # netC needs to maximize loss
                X, y = self.predicted_mask_to_full(bs)
                self.gl.append(self.model.train_on_batch(X, y))
                self.gen_iterations += 1

                if i % 10 == 0:
                    tqdm.write(
                        'G_Iters: {}, Loss_D: {:06.2f}, Loss_D_real: {:06.2f}, Loss_D_fake: {:06.2f}, Loss_G: {:06.2f} \n'.format(
                            self.gen_iterations, self.dl[-1], self.rl[-1], self.fl[-1], self.gl[-1]))

        return self.dl, self.gl, self.rl, self.fl

    def show_masks(self, out_layer=-2):
        return Model(self.netS.input, self.netS.layers[out_layer].output)