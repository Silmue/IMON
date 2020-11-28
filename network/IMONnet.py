class IMON_3(Network):
    def __init__(self, name, flow_multiplier=1., channels=8, ipmethod=0, n_pred=6, div=2, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.ipmethod = ipmethod
        self.n_pred = n_pred
        self.reconstruction = Dense3DSpatialTransformer()
        self.div = div

    def recons(self, plane, div, dvf):
        planes = tf.unstack(plane, axis=-1)
        ch = len(planes)//div
        planes[:ch] = [tf.squeeze(self.reconstruction([tf.expand_dims(planes[i], -1), dvf]), -1) for i in range(ch)]
        return tf.stack(planes, axis=-1)

    def build(self, img1, img2):
        '''
            img1, img2, flow : tensor of shape [batch, X, Y, Z, C]
        '''
        concatImgs = tf.concat([img1, img2], 4, 'concatImgs')
        dims = 3
        c = self.channels
        convs = [[], [convolveLeakyReLU('conv1', concatImgs, c,   3, 1)]]
        shapes = [concatImgs.shape.as_list(), convs[1][0].shape.as_list()]
        for i in range(2, 7):
            c *= 2
            conv = [convolveLeakyReLU('conv%d'%i, convs[-1][-1], c, 3, 2)]
            if i>2:
                conv.append(convolveLeakyReLU('conv%d_1'%i, conv[-1], c, 3, 1))
            convs.append(conv)
            shapes.append(conv[0].shape.as_list())
        concat = convs[-1][-1]

        for i in range(6, 0, -1):
            if i<6:
                deconv = upconvolveLeakyReLU('deconv%d' % i, concat, shapes[i][4], 4, 2, shapes[i][1:4])
                concat = tf.concat([convs[i][-1], deconv], 4, 'concat%d' % i)
                concat = convolveLeakyReLU('deconv%d_2' % i, concat, shapes[i][4], 3, 1)
            if self.n_pred >= i:
                pred_inc = convolve('pred%d' % i, concat, dims, 3, 1)
                if self.n_pred == i:
                    pred = pred_inc
                else:
                    pred += pred_inc
                if i>1:
                    concat = self.recons(concat, self.div, pred_inc//(2**(i+1)))
                    pred = upsample(pred, 2, self.ipmethod)
                    convs[i-1].append(self.recons(convs[i - 1][-1], self.div, pred // (2 ** i)))

        return {'flow': pred * self.flow_multiplier}


class IMON_2(Network):
    def __init__(self, name, flow_multiplier=1., channels=16, ipmethod=0, n_pred=4, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.ipmethod = ipmethod
        self.n_pred = n_pred
        self.reconstruction = Dense3DSpatialTransformer()

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
        if self.n_pred>=5:
            conv5_1 = self.reconstruction([conv5_1, upsample(pred6, upsamplescale=2, ipmethod=self.ipmethod)])
        concat5 = tf.concat([conv5_1, deconv5, upsamp6to5], 4, 'concat5')

        pred5 = convolve('pred5', concat5, dims, 3, 1)
        upsamp5to4 = upconvolve('upsamp5to4', pred5, dims, 4, 2, shape4[1:4])
        deconv4 = upconvolveLeakyReLU(
            'deconv4', concat5, shape4[4], 4, 2, shape4[1:4])
        if self.n_pred>=4:
            print(conv4_1.shape)
            conv4_1 = self.reconstruction([conv4_1, upsample(pred5, upsamplescale=2, ipmethod=self.ipmethod)])
            print(conv4_1.shape)
        concat4 = tf.concat([conv4_1, deconv4, upsamp5to4],
                            4, 'concat4')  # channel = 512+256+2

        pred4 = convolve('pred4', concat4, dims, 3, 1)
        upsamp4to3 = upconvolve('upsamp4to3', pred4, dims, 4, 2, shape3[1:4])
        deconv3 = upconvolveLeakyReLU(
            'deconv3', concat4, shape3[4], 4, 2, shape3[1:4])
        if self.n_pred>=3:
            conv3_1 = self.reconstruction([conv3_1, upsample(pred4, upsamplescale=2, ipmethod=self.ipmethod)])
        concat3 = tf.concat([conv3_1, deconv3, upsamp4to3],
                            4, 'concat3')  # channel = 256+128+2

        pred3 = convolve('pred3', concat3, dims, 3, 1)
        upsamp3to2 = upconvolve('upsamp3to2', pred3, dims, 4, 2, shape2[1:4])
        deconv2 = upconvolveLeakyReLU(
            'deconv2', concat3, shape2[4], 4, 2, shape2[1:4])
        if self.n_pred>=2:
            conv2 = self.reconstruction([conv2, upsample(pred3, upsamplescale=2, ipmethod=self.ipmethod)])
        concat2 = tf.concat([conv2, deconv2, upsamp3to2],
                            4, 'concat2')  # channel = 128+64+2

        pred2 = convolve('pred2', concat2, dims, 3, 1)
        upsamp2to1 = upconvolve('upsamp2to1', pred2, dims, 4, 2, shape1[1:4])
        deconv1 = upconvolveLeakyReLU(
            'deconv1', concat2, shape1[4], 4, 2, shape1[1:4])
        if self.n_pred>=1:
            conv1 = self.reconstruction([conv1, upsample(pred2, upsamplescale=2, ipmethod=self.ipmethod)])
        concat1 = tf.concat([conv1, deconv1, upsamp2to1], 4, 'concat1')
        pred0 = upconvolve('upsamp1to0', concat1, dims, 4, 2, shape0[1:4])
        pred = pred0
        for i in range(2, self.n_pred+1):
            pred += upsample(eval('pred{}'.format(i)), 2**i, self.ipmethod)

        return {'flow': pred * 20 * self.flow_multiplier}



class IMON(Network):
    def __init__(self, name, flow_multiplier=1., channels=16, ipmethod=0, n_pred=4, **kwargs):
        super().__init__(name, **kwargs)
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.ipmethod = ipmethod
        self.n_pred = n_pred
        

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
        pred = pred0
        for i in range(2, self.n_pred+1):
            pred += upsample(eval('pred{}'.format(i)), 2**i, self.ipmethod)

        return {'flow': pred * 20 * self.flow_multiplier}