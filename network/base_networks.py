import tensorflow as tf
import tflearn
import numpy as np
from keras.layers.convolutional import UpSampling3D
from tflearn.initializations import normal
from .spatial_transformer import Dense3DSpatialTransformer, Fast3DTransformer
from .utils import Network, ReLU, LeakyReLU

defautl_channel = 8


def convolve(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False, weights_init='uniform_scaling', regularizer=None):
    return tflearn.layers.conv_3d(inputLayer, outputChannel, kernelSize, strides=stride,
                                  padding='same', activation='linear', bias=True, scope=opName, reuse=reuse, weights_init=weights_init, regularizer=regularizer)


def convolveReLU(opName, inputLayer, outputChannel, kernelSize, stride=1, stddev=1e-2, reuse=False, regularizer=None):
    return ReLU(convolve(opName, inputLayer,
                         outputChannel,
                         kernelSize, stride, stddev=stddev, reuse=reuse, regularizer=regularizer),
                opName+'_rectified')


def convolveLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride=1, alpha=0.1, stddev=1e-2, reuse=False, regularizer=None):
    return LeakyReLU(convolve(opName, inputLayer,
                              outputChannel,
                              kernelSize, stride, stddev, reuse, regularizer=regularizer),
                     alpha, opName+'_leakilyrectified')


def upconvolve(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, stddev=1e-2, reuse=False, weights_init='uniform_scaling', regularizer=None):
    return tflearn.layers.conv.conv_3d_transpose(inputLayer, outputChannel, kernelSize, targetShape, strides=stride,
                                                 padding='same', activation='linear', bias=False, scope=opName, reuse=reuse, weights_init=weights_init, regularizer=regularizer)


def upconvolveReLU(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, stddev=1e-2, reuse=False, regularizer=None):
    return ReLU(upconvolve(opName, inputLayer,
                           outputChannel,
                           kernelSize, stride,
                           targetShape, stddev, reuse, regularizer=regularizer),
                opName+'_rectified')


def upconvolveLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, alpha=0.1, stddev=1e-2, reuse=False, regularizer=None):
    return LeakyReLU(upconvolve(opName, inputLayer,
                                outputChannel,
                                kernelSize, stride,
                                targetShape, stddev, reuse, regularizer=regularizer),
                     alpha, opName+'_rectified')


def upsample(input_, upsamplescale, ipmethod, shape):
    bs, xdim, ydim, zdim, channel_count = shape
    channel_count = 3
    xdim *= upsamplescale
    ydim *= upsamplescale
    zdim *= upsamplescale
    bz = tf.shape(input_)[0]
    deconv = tf.nn.conv3d_transpose(value=input_, filter=tf.ones([upsamplescale, upsamplescale, upsamplescale, channel_count, channel_count], tf.float32), output_shape=[bz, xdim, ydim, zdim, channel_count],
                                    strides=[1, upsamplescale,
                                             upsamplescale, upsamplescale, 1],
                                    padding="SAME", name='UpsampleDeconv')
    if ipmethod == 1:
        smooth5d = tf.constant(np.ones([upsamplescale, upsamplescale, upsamplescale, channel_count, channel_count], dtype='float32')/np.float32(
            upsamplescale)/np.float32(upsamplescale)/np.float32(upsamplescale), name='Upsample'+str(upsamplescale))
        # print('Upsample', upsamplescale)
        return tf.nn.conv3d(input=deconv,
                            filter=smooth5d,
                            strides=[1, 1, 1, 1, 1],
                            padding='SAME',
                            name='UpsampleSmooth'+str(upsamplescale))
    else:
        return deconv


def encoderBlock(opName, inputLayer, outputChannel, kernelSize, stride, alpha=0.1, stddev=1e-2, reuse=False):
    intputChannel = inputLayer.shape.as_list()[-1]
    residual = inputLayer
    x = convolveLeakyReLU(opName+"_1", inputLayer, outputChannel, kernelSize, stride, alpha=alpha, stddev=stddev, reuse=reuse)
    x = convolve(opName+"_2", x, outputChannel, kernelSize, 1, stddev=stddev, reuse=reuse)
    if not (intputChannel==outputChannel and stride==1):
        residual = convolve(opName+"_1X1", residual, outputChannel, 1, stride)
    return LeakyReLU(tf.add(x, residual), alpha, opName+'_leakilyrectified')


class SiameseLink(Network):
    def __init__(self, name, flow_multiplier=1., channels=defautl_channel, ipmethod=0, depth=5, n_pred=3, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.ipmethod = ipmethod
        self.depth = depth
        self.n_pred = n_pred
        self.reconstruction = Dense3DSpatialTransformer()

    def build(self, imgf, imgm):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''

        dims = 3
        c = self.channels
        # 16 * 32 = 512 channels
        convf = [convolveLeakyReLU('conv0', imgf, c, 3, 1)]
        convm = [convolveLeakyReLU('conv0', imgm, c, 3, 1, reuse=True)]
        shapes = [convf[-1].shape.as_list()]
        for i in range(1, self.depth+1):
            convf.append(encoderBlock(
                'conv{}'.format(i), convf[-1], c << i, 3, 2))
            convm.append(encoderBlock(
                'conv{}'.format(i), convm[-1], c << i, 3, 2, reuse=True))
            shapes.append(convf[-1].shape.as_list())
        concat = [tf.concat([convf[-1], convm[-1]], 4,
                            'concat{}'.format(self.depth))]
        iconv = []
        pred = []
        pred_inc = []
        deconv = []
        # print('-------------------\n conv shape:{}\n----------------------\n'.format(shapes))

        for i in range(self.depth, 0, -1):
            iconv.append(encoderBlock('iconv{}'.format(i), concat[-1], shapes[i][-1], 3, 1))
            # iconv.append(encoderBlock('iconv{}_1'.format(i), concat[-1], shapes[i][-1], 3, 1))
            # pred_inc.append(upconvolveLeakyReLU('pred_inc{}'.format(i), iconv[-1], 3, 4, 2, shapes[i-1][1:-1]))
            # pred.append(self.reconstruction([pred_inc[-1], UpSampling3D()(pred[-1]*2)]) if len(pred) else pred_inc[-1])
            deconv.append(upconvolveLeakyReLU('deconv{}'.format(
                i-1), iconv[-1], shapes[i-1][-1], 4, 2, shapes[i-1][1:-1]))
            if i <= self.n_pred:
                pred_inc.append(convolveLeakyReLU(
                    'pred_inc{}'.format(i), deconv[-1], 3, 3, 1)*(1 << i))
                pred.append(pred_inc[-1]+self.reconstruction([UpSampling3D()(pred[-1]*2),
                                                              pred_inc[-1]]) if len(pred) else pred_inc[-1])
                concat.append(tf.concat([deconv[-1], self.reconstruction(
                    [convm[i-1], pred[-1]]), convf[i-1]], 4, 'concat{}'.format(i-1)))
            else:
                concat.append(
                    tf.concat([deconv[-1], convm[i-1], convf[i-1]], 4, 'concat{}'.format(i-1)))
        iconv.append(encoderBlock(
            'iconv0', concat[-1], shapes[0][-1], 3, 1))
        # iconv.append(encoderBlock(
        #     'iconv0_1', concat[-1], shapes[0][-1], 3, 1))
        pred_inc.append(convolveLeakyReLU('pred_inc0', iconv[-1], 3, 3, 1))
        pred[-1] = pred_inc[-1]+self.reconstruction([pred[-1], pred_inc[-1]])
        # print('-------------------\n deconv shape:{}\n----------------------\n'.format([x.shape.as_list() for x in deconv]))
        # print('-------------------\n pred shape:{}\n----------------------\n'.format([x.shape.as_list() for x in pred]))
        # print('-------------------\n predinc shape:{}\n----------------------\n'.format([x.shape.as_list() for x in pred_inc]))

        return {'flow': pred[-1] * self.flow_multiplier,
                'pred_inc': pred_inc}


class Siamese2conv(Network):
    def __init__(self, name, flow_multiplier=1., channels=defautl_channel, depth=5, n_pred=3, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.depth = depth
        self.n_pred = n_pred
        self.reconstruction = Dense3DSpatialTransformer()

    def build(self, imgf, imgm):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''

        dims = 3
        c = self.channels
        # 16 * 32 = 512 channels
        convf = [convolveLeakyReLU('conv0', imgf, c, 3, 1)]
        convm = [convolveLeakyReLU('conv0', imgm, c, 3, 1, reuse=True)]
        shapes = [convf[-1].shape.as_list()]
        for i in range(1, self.depth+1):
            convf.append(convolveLeakyReLU(
                'conv{}'.format(i), convf[-1], c << i, 3, 2))
            convm.append(convolveLeakyReLU(
                'conv{}'.format(i), convm[-1], c << i, 3, 2, reuse=True))
            convf = convolveLeakyReLU(
                'conv{}_1'.format(i), convf[-1], c << i, 3)
            convm = convolveLeakyReLU(
                'conv{}_1'.format(i), convm[-1], c << i, 3, reuse=True)
            shapes.append(convf[-1].shape.as_list())
        concat = [tf.concat([convf[-1], convm[-1]], 4,
                            'concat{}'.format(self.depth))]
        iconv = []
        pred = []
        pred_inc = []
        deconv = []
        # print('-------------------\n conv shape:{}\n----------------------\n'.format(shapes))

        for i in range(self.depth, 0, -1):
            iconv.append(convolveLeakyReLU('iconv{}_0'.format(i),
                                           concat[-1], shapes[i][-1], 3, 1))
            iconv.append(convolveLeakyReLU('iconv{}_1'.format(i),
                                           iconv[-1], shapes[i][-1], 3, 1))
            # pred_inc.append(upconvolveLeakyReLU('pred_inc{}'.format(i), iconv[-1], 3, 4, 2, shapes[i-1][1:-1]))
            # pred.append(self.reconstruction([pred_inc[-1], UpSampling3D()(pred[-1]*2)]) if len(pred) else pred_inc[-1])
            deconv.append(upconvolveLeakyReLU('deconv{}'.format(
                i-1), iconv[-1], shapes[i-1][-1], 4, 2, shapes[i-1][1:-1]))
            if i < self.n_pred:
                pred_inc.append(convolveLeakyReLU(
                    'pred_inc{}'.format(i), deconv[-1], 3, 3, 1)*(1 << i))
                pred.append(pred_inc[-1]+self.reconstruction([UpSampling3D()(pred[-1]*2),
                                                              pred_inc[-1]]) if len(pred) else pred_inc[-1])
                concat.append(tf.concat([deconv[-1], self.reconstruction(
                    [convm[i-1], pred[-1]]), convf[i-1]], 4, 'concat{}'.format(i-1)))
            else:
                concat.append(
                    tf.concat([deconv[-1], convm[i-1], convf[i-1]], 4, 'concat{}'.format(i-1)))
        iconv.append(convolveLeakyReLU(
            'iconv0_0', concat[-1], shapes[0][-1], 3, 1))
        iconv.append(convolveLeakyReLU(
            'iconv0_1', iconv[-1], shapes[0][-1], 3, 1))
        pred_inc.append(convolveLeakyReLU('pred_inc0', iconv[-1], 3, 3, 1))
        pred[-1] = pred_inc[-1]+self.reconstruction([pred[-1], pred_inc[-1]])
        # print('-------------------\n deconv shape:{}\n----------------------\n'.format([x.shape.as_list() for x in deconv]))
        # print('-------------------\n pred shape:{}\n----------------------\n'.format([x.shape.as_list() for x in pred]))
        # print('-------------------\n predinc shape:{}\n----------------------\n'.format([x.shape.as_list() for x in pred_inc]))

        return {'flow': pred[-1] * self.flow_multiplier,
                'pred_inc': pred_inc}


class Siamese(Network):
    def __init__(self, name, flow_multiplier=1., channels=defautl_channel, depth=5, n_pred=3, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.depth = depth
        self.n_pred = n_pred
        self.reconstruction = Dense3DSpatialTransformer()

    def build(self, imgf, imgm):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''

        dims = 3
        c = self.channels
        # 16 * 32 = 512 channels
        convf = [convolveLeakyReLU('conv0', imgf, c, 3, 1)]
        convm = [convolveLeakyReLU('conv0', imgm, c, 3, 1, reuse=True)]
        shapes = [convf[-1].shape.as_list()]
        for i in range(1, self.depth+1):
            convf.append(convolveLeakyReLU(
                'conv{}'.format(i), convf[-1], c << i, 3, 2))
            convm.append(convolveLeakyReLU(
                'conv{}'.format(i), convm[-1], c << i, 3, 2, reuse=True))
            shapes.append(convf[-1].shape.as_list())
        concat = [tf.concat([convf[-1], convm[-1]], 4,
                            'concat{}'.format(self.depth))]
        iconv = []
        pred = []
        pred_inc = []
        deconv = []
        # print('-------------------\n conv shape:{}\n----------------------\n'.format(shapes))

        for i in range(self.depth, 0, -1):
            iconv.append(convolveLeakyReLU('iconv{}_0'.format(i),
                                           concat[-1], shapes[i][-1], 3, 1))
            iconv.append(convolveLeakyReLU('iconv{}_1'.format(i),
                                           concat[-1], shapes[i][-1], 3, 1))
            # pred_inc.append(upconvolveLeakyReLU('pred_inc{}'.format(i), iconv[-1], 3, 4, 2, shapes[i-1][1:-1]))
            # pred.append(self.reconstruction([pred_inc[-1], UpSampling3D()(pred[-1]*2)]) if len(pred) else pred_inc[-1])
            deconv.append(upconvolveLeakyReLU('deconv{}'.format(
                i-1), iconv[-1], shapes[i-1][-1], 4, 2, shapes[i-1][1:-1]))
            if i < self.n_pred:
                pred_inc.append(convolveLeakyReLU(
                    'pred_inc{}'.format(i), deconv[-1], 3, 3, 1)*(1 << i))
                pred.append(pred_inc[-1]+self.reconstruction([UpSampling3D()(pred[-1]*2),
                                                              pred_inc[-1]]) if len(pred) else pred_inc[-1])
                concat.append(tf.concat([deconv[-1], self.reconstruction(
                    [convm[i-1], pred[-1]]), convf[i-1]], 4, 'concat{}'.format(i-1)))
            else:
                concat.append(
                    tf.concat([deconv[-1], convm[i-1], convf[i-1]], 4, 'concat{}'.format(i-1)))
        iconv.append(convolveLeakyReLU(
            'iconv0_0', concat[-1], shapes[0][-1], 3, 1))
        iconv.append(convolveLeakyReLU(
            'iconv0_1', concat[-1], shapes[0][-1], 3, 1))
        pred_inc.append(convolveLeakyReLU('pred_inc0', iconv[-1], 3, 3, 1))
        pred[-1] = pred_inc[-1]+self.reconstruction([pred[-1], pred_inc[-1]])
        # print('-------------------\n deconv shape:{}\n----------------------\n'.format([x.shape.as_list() for x in deconv]))
        # print('-------------------\n pred shape:{}\n----------------------\n'.format([x.shape.as_list() for x in pred]))
        # print('-------------------\n predinc shape:{}\n----------------------\n'.format([x.shape.as_list() for x in pred_inc]))

        return {'flow': pred[-1] * self.flow_multiplier,
                'pred_inc': pred_inc}


class VTN(Network):
    def __init__(self, name, flow_multiplier=1., channels=defautl_channel, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels

    def build(self, img1, img2):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        concatImgs = tf.concat([img1, img2], 4, 'concatImgs')

        dims = 3
        c = self.channels
        conv1 = convolveLeakyReLU(
            'conv1',   concatImgs, c,   3, 2)  # 64 * 64 * 64
        conv2 = convolveLeakyReLU(
            'conv2',   conv1,      c*2,   3, 2)  # 32 * 32 * 32
        conv3 = convolveLeakyReLU('conv3',   conv2,      c*4,   3, 2)
        conv3_1 = convolveLeakyReLU('conv3_1', conv3,      c*4,   3, 1)
        conv4 = convolveLeakyReLU(
            'conv4',   conv3_1,    c*8,  3, 2)  # 16 * 16 * 16
        conv4_1 = convolveLeakyReLU('conv4_1', conv4,      c*8,  3, 1)
        conv5 = convolveLeakyReLU(
            'conv5',   conv4_1,    c*16,  3, 2)  # 8 * 8 * 8
        conv5_1 = convolveLeakyReLU('conv5_1', conv5,      c*16,  3, 1)
        conv6 = convolveLeakyReLU(
            'conv6',   conv5_1,    c*32,  3, 2)  # 4 * 4 * 4
        conv6_1 = convolveLeakyReLU('conv6_1', conv6,      c*32,  3, 1)
        # 16 * 32 = 512 channels

        shape0 = concatImgs.shape.as_list()
        shape1 = conv1.shape.as_list()
        shape2 = conv2.shape.as_list()
        shape3 = conv3.shape.as_list()
        shape4 = conv4.shape.as_list()
        shape5 = conv5.shape.as_list()
        shape6 = conv6.shape.as_list()

        pred6 = convolve('pred6', conv6_1, dims, 3, 1)
        upsamp6to5 = upconvolve('upsamp6to5', pred6, dims, 4, 2, shape5[1:4])
        deconv5 = upconvolveLeakyReLU(
            'deconv5', conv6_1, shape5[4], 4, 2, shape5[1:4])
        concat5 = tf.concat([conv5_1, deconv5, upsamp6to5], 4, 'concat5')

        pred5 = convolve('pred5', concat5, dims, 3, 1)
        upsamp5to4 = upconvolve('upsamp5to4', pred5, dims, 4, 2, shape4[1:4])
        deconv4 = upconvolveLeakyReLU(
            'deconv4', concat5, shape4[4], 4, 2, shape4[1:4])
        concat4 = tf.concat([conv4_1, deconv4, upsamp5to4],
                            4, 'concat4')  # channel = 512+256+2

        pred4 = convolve('pred4', concat4, dims, 3, 1)
        upsamp4to3 = upconvolve('upsamp4to3', pred4, dims, 4, 2, shape3[1:4])
        deconv3 = upconvolveLeakyReLU(
            'deconv3', concat4, shape3[4], 4, 2, shape3[1:4])
        concat3 = tf.concat([conv3_1, deconv3, upsamp4to3],
                            4, 'concat3')  # channel = 256+128+2

        pred3 = convolve('pred3', concat3, dims, 3, 1)
        upsamp3to2 = upconvolve('upsamp3to2', pred3, dims, 4, 2, shape2[1:4])
        deconv2 = upconvolveLeakyReLU(
            'deconv2', concat3, shape2[4], 4, 2, shape2[1:4])
        concat2 = tf.concat([conv2, deconv2, upsamp3to2],
                            4, 'concat2')  # channel = 128+64+2

        pred2 = convolve('pred2', concat2, dims, 3, 1)
        upsamp2to1 = upconvolve('upsamp2to1', pred2, dims, 4, 2, shape1[1:4])
        deconv1 = upconvolveLeakyReLU(
            'deconv1', concat2, shape1[4], 4, 2, shape1[1:4])
        concat1 = tf.concat([conv1, deconv1, upsamp2to1], 4, 'concat1')
        pred0 = upconvolve('upsamp1to0', concat1, dims, 4, 2, shape0[1:4])

        return {'flow': pred0 * 20 * self.flow_multiplier}


class VoxelMorph(Network):
    def __init__(self, name, flow_multiplier=1., channels=defautl_channel, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.encoders = [m * channels for m in [1, 2, 2, 2]]
        self.decoders = [m * channels for m in [2, 2, 2, 2, 2, 1, 1]] + [3]

    def build(self, img1, img2):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        concatImgs = tf.concat([img1, img2], 4, 'concatImgs')

        conv1 = convolveLeakyReLU(
            'conv1',   concatImgs, self.encoders[0],     3, 2)  # 64 * 64 * 64
        conv2 = convolveLeakyReLU(
            'conv2',   conv1,      self.encoders[1],   3, 2)  # 32 * 32 * 32
        conv3 = convolveLeakyReLU(
            'conv3',   conv2,      self.encoders[2],   3, 2)  # 16 * 16 * 16
        conv4 = convolveLeakyReLU(
            'conv4',   conv3,      self.encoders[3],   3, 2)  # 8 * 8 * 8

        net = convolveLeakyReLU('decode4', conv4, self.decoders[0], 3, 1)
        net = tf.concat([UpSampling3D()(net), conv3], axis=-1)
        net = convolveLeakyReLU('decode3',   net, self.decoders[1], 3, 1)
        net = tf.concat([UpSampling3D()(net), conv2], axis=-1)
        net = convolveLeakyReLU('decode2',   net, self.decoders[2], 3, 1)
        net = tf.concat([UpSampling3D()(net), conv1], axis=-1)
        net = convolveLeakyReLU('decode1',   net, self.decoders[3], 3, 1)
        net = convolveLeakyReLU('decode1_1', net, self.decoders[4], 3, 1)
        net = tf.concat([UpSampling3D()(net), concatImgs], axis=-1)
        net = convolveLeakyReLU('decode0',   net, self.decoders[5], 3, 1)
        if len(self.decoders) == 8:
            net = convolveLeakyReLU('decode0_1', net, self.decoders[6], 3, 1)
        net = convolve(
            'flow', net, self.decoders[-1], 3, 1, weights_init=normal(stddev=1e-5))
        return {
            'flow': net * self.flow_multiplier
        }


def affine_flow(W, b, len1, len2, len3):
    b = tf.reshape(b, [-1, 1, 1, 1, 3])
    xr = tf.range(-(len1 - 1) / 2.0, len1 / 2.0, 1.0, tf.float32)
    xr = tf.reshape(xr, [1, -1, 1, 1, 1])
    yr = tf.range(-(len2 - 1) / 2.0, len2 / 2.0, 1.0, tf.float32)
    yr = tf.reshape(yr, [1, 1, -1, 1, 1])
    zr = tf.range(-(len3 - 1) / 2.0, len3 / 2.0, 1.0, tf.float32)
    zr = tf.reshape(zr, [1, 1, 1, -1, 1])
    wx = W[:, :, 0]
    wx = tf.reshape(wx, [-1, 1, 1, 1, 3])
    wy = W[:, :, 1]
    wy = tf.reshape(wy, [-1, 1, 1, 1, 3])
    wz = W[:, :, 2]
    wz = tf.reshape(wz, [-1, 1, 1, 1, 3])
    return (xr * wx + yr * wy) + (zr * wz + b)


def det3x3(M):
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    return tf.add_n([
        M[0][0] * M[1][1] * M[2][2],
        M[0][1] * M[1][2] * M[2][0],
        M[0][2] * M[1][0] * M[2][1]
    ]) - tf.add_n([
        M[0][0] * M[1][2] * M[2][1],
        M[0][1] * M[1][0] * M[2][2],
        M[0][2] * M[1][1] * M[2][0]
    ])


class VTNAffineStem(Network):
    def __init__(self, name, flow_multiplier=1., **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier

    def build(self, img1, img2):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        concatImgs = tf.concat([img1, img2], 4, 'coloncatImgs')

        dims = 3
        conv1 = convolveLeakyReLU(
            'conv1',   concatImgs, 16,   3, 2)  # 64 * 64 * 64
        conv2 = convolveLeakyReLU(
            'conv2',   conv1,      32,   3, 2)  # 32 * 32 * 32
        conv3 = convolveLeakyReLU('conv3',   conv2,      64,   3, 2)
        conv3_1 = convolveLeakyReLU(
            'conv3_1', conv3,      64,   3, 1)
        conv4 = convolveLeakyReLU(
            'conv4',   conv3_1,    128,  3, 2)  # 16 * 16 * 16
        conv4_1 = convolveLeakyReLU(
            'conv4_1', conv4,      128,  3, 1)
        conv5 = convolveLeakyReLU(
            'conv5',   conv4_1,    256,  3, 2)  # 8 * 8 * 8
        conv5_1 = convolveLeakyReLU(
            'conv5_1', conv5,      256,  3, 1)
        conv6 = convolveLeakyReLU(
            'conv6',   conv5_1,    512,  3, 2)  # 4 * 4 * 4
        conv6_1 = convolveLeakyReLU(
            'conv6_1', conv6,      512,  3, 1)
        ks = conv6_1.shape.as_list()[1:4]
        conv7_W = tflearn.layers.conv_3d(
            conv6_1, 9, ks, strides=1, padding='valid', activation='linear', bias=False, scope='conv7_W')
        conv7_b = tflearn.layers.conv_3d(
            conv6_1, 3, ks, strides=1, padding='valid', activation='linear', bias=False, scope='conv7_b')

        I = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        W = tf.reshape(conv7_W, [-1, 3, 3]) * self.flow_multiplier
        b = tf.reshape(conv7_b, [-1, 3]) * self.flow_multiplier
        A = W + I
        # the flow is displacement(x) = place(x) - x = (Ax + b) - x
        # the model learns W = A - I.

        sx, sy, sz = img1.shape.as_list()[1:4]
        flow = affine_flow(W, b, sx, sy, sz)
        # determinant should be close to 1
        det = det3x3(A)
        det_loss = tf.nn.l2_loss(det - 1.0)
        # should be close to being orthogonal
        # C=A'A, a positive semi-definite matrix
        # should be close to I. For this, we require C
        # has eigen values close to 1 by minimizing
        # k1+1/k1+k2+1/k2+k3+1/k3.
        # to prevent NaN, minimize
        # k1+eps + (1+eps)^2/(k1+eps) + ...
        eps = 1e-5
        epsI = [[[eps * elem for elem in row] for row in Mat] for Mat in I]
        C = tf.matmul(A, A, True) + epsI

        def elem_sym_polys_of_eigen_values(M):
            M = [[M[:, i, j] for j in range(3)] for i in range(3)]
            sigma1 = tf.add_n([M[0][0], M[1][1], M[2][2]])
            sigma2 = tf.add_n([
                M[0][0] * M[1][1],
                M[1][1] * M[2][2],
                M[2][2] * M[0][0]
            ]) - tf.add_n([
                M[0][1] * M[1][0],
                M[1][2] * M[2][1],
                M[2][0] * M[0][2]
            ])
            sigma3 = tf.add_n([
                M[0][0] * M[1][1] * M[2][2],
                M[0][1] * M[1][2] * M[2][0],
                M[0][2] * M[1][0] * M[2][1]
            ]) - tf.add_n([
                M[0][0] * M[1][2] * M[2][1],
                M[0][1] * M[1][0] * M[2][2],
                M[0][2] * M[1][1] * M[2][0]
            ])
            return sigma1, sigma2, sigma3
        s1, s2, s3 = elem_sym_polys_of_eigen_values(C)
        ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
        ortho_loss = tf.reduce_sum(ortho_loss)

        return {
            'flow': flow,
            'W': W,
            'b': b,
            'det_loss': det_loss,
            'ortho_loss': ortho_loss
        }


class FullDiscriminator(Network):
    def __init__(self, name, channels=16, depth=4, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels
        self.depth = depth

    def build(self, img1, img2):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        concatImgs = tf.concat([img1, img2], 4, 'concatImgs')
        conv = []
        for i in range(self.depth-1):
            conv.append(convolveLeakyReLU('conv%d' % i, conv[-1] if len(conv) else concatImgs, self.channels << i, 3, 2))
        conv_pr = tf.sigmoid(convolve('convPr', conv[-1], 1, 3))
        pr = tf.reduce_mean(conv_pr)
        return {
            'prob': pr,
            'positive': -tf.log(tf.clip_by_value(pr, 1e-8, 1.0)),
            'negative': -tf.log(tf.clip_by_value(1-pr, 1e-8, 1.0))
        }
        # return {
        #     'prob': fc,
        #     'positive': tf.reduce_sum(1-fc),
        #     'negative': tf.reduce_sum(fc)
        # }

 
class FeatureExtractor(Network):
    def __init__(self, name, channels=16, depth=3, **kwargs):
        super().__init__(name, **kwargs)
        self.channels = channels
        self.depth = depth

    def build(self, img):
        conv = []
        shapes = [img.shape.as_list()]
        shapes[0][-1] = self.channels
        for i in range(self.depth):
            conv.append(convolveLeakyReLU('fconv%d' % i, conv[-1] if len(conv) else img, self.channels << i, 3, 2, regularizer='L2'))
            shapes.append(conv[-1].shape.as_list())
        # print(shapes)
        concat = []
        deconv = []
        for i in range(self.depth-1, -1, -1):
            concat.append(tf.concat([deconv[-1], conv[i]], axis=4) if len(deconv) else conv[i])
            deconv.append(upconvolveLeakyReLU('fdeconv%d' % i, concat[-1], shapes[i][4], 4, 2, [128>>i]*3, regularizer='L2'))
        out = convolveLeakyReLU('conv_out', tf.concat([deconv[-1], img], axis=4), self.channels, 3, regularizer='L2')
        return out


class PartDiscriminator(Network):
    def __init__(self, name, loss='l2', channels=16, **kwargs):
        super().__init__(name, **kwargs)
        self.feature = FeatureExtractor('feature')
        self.loss = loss

    @property
    def trainable_variables(self):
        return self.feature.trainable_variables

    @property
    def l2_regularizer(self, decay=0.1):
        return decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in self.trainable_variables if 'b:0' not in v.name])

    def build(self, img1, img2):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        f1 = self.feature(img1)
        f2 = self.feature(img2)
        if self.loss=='cc':
            cc = correlation_coefficient(f1, f2)
            return {
                'prob' : tf.reduce_sum(cc),
                'positive': tf.reduce_sum(1-cc),
                'negative': tf.reduce_sum(cc)
            }
        else:
            l2 = tf.nn.l2_loss(f1-f2)/(128**3)
            return {
                'prob' : l2,
                'positive': l2,
                'negative': -l2
            }


def correlation_coefficient(img1, warped_img2):
    sizes = np.prod(img1.shape.as_list()[1:])
    flatten1 = tf.reshape(img1, [-1, sizes])
    flatten2 = tf.reshape(warped_img2, [-1, sizes])

    mean1 = tf.reduce_mean(flatten1, axis=-1, keepdims=True)
    mean2 = tf.reduce_mean(flatten2, axis=-1, keepdims=True)
    var1 = tf.reduce_mean(tf.square(flatten1 - mean1), axis=-1)
    var2 = tf.reduce_mean(tf.square(flatten2 - mean2), axis=-1)
    cov12 = tf.reduce_mean(
        (flatten1 - mean1) * (flatten2 - mean2), axis=-1)
    pearson_r = cov12 / tf.sqrt((var1 + 1e-6) * (var2 + 1e-6))

    raw_loss = 1 - pearson_r
    raw_loss = tf.reduce_sum(raw_loss)
    return pearson_r
