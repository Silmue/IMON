import argparse
import numpy as np
import os
import json
import h5py
import copy
import collections
import re
import datetime
import hashlib
import time
from timeit import default_timer

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--base_network', type=str, default='VTN',
                    help='Specifies the base network (either VTN or VoxelMorph)')
parser.add_argument('-n', '--n_cascades', type=int, default=1,
                    help='Number of cascades')
parser.add_argument('-r', '--rep', type=int, default=1,
                    help='Number of times of shared-weight cascading')
parser.add_argument('-g', '--gpu', type=str, default='-1',
                    help='Specifies gpu device(s)')
parser.add_argument('-c', '--checkpoint', type=str, default=None,
                    help='Specifies a previous checkpoint to start with')
parser.add_argument('-d', '--dataset', type=str, default="datasets/brain.json",
                    help='Specifies a data config')
parser.add_argument('--batch', type=int, default=1,
                    help='Number of image pairs per batch') 
parser.add_argument('--fake_batch', type=int, default=1,
                    help='Number of image pairs per batch by gradient accmulate') 
parser.add_argument('--round', type=int, default=20000,
                    help='Number of batches per epoch')
parser.add_argument('--epochs', type=float, default=5,
                    help='Number of epochs')
parser.add_argument('--fast_reconstruction', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--val_steps', type=int, default=100)
parser.add_argument('--net_args', type=str, default='')
parser.add_argument('--data_args', type=str, default='')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--clear_steps', action='store_true')
parser.add_argument('--finetune', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--logs', type=str, default='')
parser.add_argument('--n_pred', type=int, default=3)
parser.add_argument('--ipmethod', type=int, default=0)
parser.add_argument('--depth', type=int, default=5)
parser.add_argument('--discriminator', type=str, default=None)
parser.add_argument('--pre_step', type=int, default=10000)
parser.add_argument('--D_step', type=int, default=0)
parser.add_argument('--loss', type=str, default='NCC')
parser.add_argument('--nccwin', type=int, default=9)
parser.add_argument('--loadmode', type=int, default=1)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import tensorflow as tf
import tflearn
import keras

import network
import data_util.liver
import data_util.brain
from data_util.data import Split

def main():
    repoRoot = os.path.dirname(os.path.realpath(__file__))

    if args.finetune is not None:
        args.clear_steps = True

    batchSize = args.batch
    iterationSize = args.round

    gpus = 0 if args.gpu == '-1' else len(args.gpu.split(','))

    Framework = network.FrameworkUnsupervised
    Framework.net_args['base_network'] = args.base_network
    Framework.net_args['n_cascades'] = args.n_cascades
    Framework.net_args['rep'] = args.rep
    Framework.net_args['n_pred'] = args.n_pred
    Framework.net_args['depth'] = args.depth
    Framework.net_args['ipmethod'] = args.ipmethod
    Framework.net_args['nccwin'] = args.nccwin
    Framework.net_args['loss'] = args.loss
    Framework.net_args.update(eval('dict({})'.format(args.net_args)))
    with open(os.path.join(args.dataset), 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [128, 128, 128])
        image_type = cfg.get('image_type')
    framework = Framework(devices=gpus, image_size=image_size, segmentation_class_value=cfg.get('segmentation_class_value', None), fast_reconstruction = args.fast_reconstruction, discriminator=args.discriminator)
    Dataset = eval('data_util.{}.Dataset'.format(image_type))
    print('Graph built.')

    # load training set and validation set

    def set_tf_keys(feed_dict, **kwargs):
        ret = dict([(k + ':0', v) if type(k)==str else (k,v) for k, v in feed_dict.items()])
        ret.update([(k + ':0', v) for k, v in kwargs.items()])
        return ret

    config = tf.ConfigProto()
    # config.intra_op_parallelism_threads = 8
    # config.inter_op_parallelism_threads = 8 
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=5, keep_checkpoint_every_n_hours=5)
        if args.checkpoint is None:
            steps = 0
            tf.global_variables_initializer().run()
        else:
            if '\\' not in args.checkpoint and '/' not in args.checkpoint:
                args.checkpoint = os.path.join(
                    repoRoot, 'weights', args.checkpoint)
            if os.path.isdir(args.checkpoint):
                args.checkpoint = tf.train.latest_checkpoint(args.checkpoint)

            tf.global_variables_initializer().run()
            checkpoints = args.checkpoint.split(';')
            var = tf.global_variables()
            # Rvar = [val for val in var if 'frm' in val.name]
            # Dvar = [val for val in var if 'frm' not in val.name]
            Rvar = [val for val in var if 'feat' not in val.name and 'frm' in val.name]
            Dvar = [val for val in var if 'feat' in val.name or 'frm' not in val.name]
            varlist = Rvar if args.loadmode==1 else (Dvar if args.loadmode==2 else var)
            if args.clear_steps:
                steps = 0
            else:
                steps = int(re.search('model-(\d+)', checkpoints[0]).group(1))
            for cp in checkpoints:
                saver = tf.train.Saver(varlist)
                saver.restore(sess, cp)

        affcnt = 0
        defcnt = 0
        for v in tf.trainable_variables():
            print(v)
            if 'affine' in v.name:
                affcnt += np.prod(v.get_shape().as_list())
            elif 'deform' in v.name and 'feat' not in v.name:
                defcnt += np.prod(v.get_shape().as_list())

        print("-------\n trainable varibales: affine %d, deform %d\n----------\n" %(affcnt, defcnt))



        data_args = eval('dict({})'.format(args.data_args))
        data_args.update(framework.data_args)
        print('data_args', data_args)
        dataset = Dataset(args.dataset, **data_args)
        if args.finetune is not None:
            if 'finetune-train-%s' % args.finetune in dataset.schemes:
                dataset.schemes[Split.TRAIN] = dataset.schemes['finetune-train-%s' %
                                                               args.finetune]
            if 'finetune-val-%s' % args.finetune in dataset.schemes:
                dataset.schemes[Split.VALID] = dataset.schemes['finetune-val-%s' %
                                                               args.finetune]
            print('train', dataset.schemes[Split.TRAIN])
            print('val', dataset.schemes[Split.VALID])
        generator = dataset.generator(Split.TRAIN, batch_size=batchSize, loop=True)

        if not args.debug:
            if args.finetune is not None:
                run_id = os.path.basename(os.path.dirname(args.checkpoint))
                if not run_id.endswith('_ft' + args.finetune):
                    run_id = run_id + '_ft' + args.finetune
            else:
                pad = ''
                retry = 1
                while True:
                    dt = datetime.datetime.now(
                        tz=datetime.timezone(datetime.timedelta(hours=8)))
                    run_id = dt.strftime('%b%d-%H%M') + pad
                    modelPrefix = os.path.join(repoRoot, 'weights', run_id)
                    try:
                        os.makedirs(modelPrefix)
                        break
                    except Exception as e:
                        print('Conflict with {}! Retry...'.format(run_id))
                        pad = '_{}'.format(retry)
                        retry += 1
            modelPrefix = os.path.join(repoRoot, 'weights', run_id)
            if not os.path.exists(modelPrefix):
                os.makedirs(modelPrefix)
            if args.name is not None:
                run_id += '_' + args.name
            if args.logs is None:
                log_dir = 'logs'
            else:
                log_dir = os.path.join('logs', args.logs)
            summary_path = os.path.join(repoRoot, log_dir, run_id)
            if not os.path.exists(summary_path):
                os.makedirs(summary_path)
            summaryWriter = tf.summary.FileWriter(summary_path, sess.graph)
            with open(os.path.join(modelPrefix, 'args.json'), 'w') as fo:
                json.dump(vars(args), fo)

        if args.finetune is not None:
            learningRates = [1e-5 / 2, 1e-5 / 2, 1e-5 / 2, 1e-5 / 4, 1e-5 / 8]
            #args.epochs = 1
        else:
            learningRates = [1e-4, 1e-4, 1e-4, 1e-4 / 2, 1e-4 / 4,
                               1e-4 / 8, 1e-4 / 16, 1e-4 / 32, 1e-4 / 64]

            # Training

        def get_lr(steps):
            m = args.lr / learningRates[0]
            return m * learningRates[min(steps // (4*iterationSize), len(learningRates)-1)]

        last_save_stamp = time.time()

        def average_gradients(grads, var):
            ret = {}
            # for grad_list in zip(*grads):
            #     grad, var = grad_list[0]
            #     if grad is not None:
            #         ret.append(
            #             (sum([grad for grad, _ in grad_list]) / len(grad_list), var))
            for i in range(len(var)):
                ret[var[i]] = sum([grad[i] for grad in grads])
            return ret

        def update_step(k, summopt=None, **kwargs):
            opt = framework.O[k]
            grad = framework.G[k]
            var = framework.V[k]
            t1 = 0
            grads = []
            summ = None
            for i in range(args.fake_batch):
                t0 = default_timer()
                fd = next(generator)
                fd.pop('mask', [])
                fd.pop('id1', [])
                fd.pop('id2', [])
                t1 += default_timer()-t0
                if summopt is not None and i==0:
                    summ, _g = sess.run([summopt, grad], set_tf_keys(fd))
                else:
                    _g = sess.run(grad, set_tf_keys(fd))
                grads.append(_g)

            grads_mean = average_gradients(grads, var)
            _ = sess.run(opt, set_tf_keys(grads_mean, **kwargs))
            return summ, t1


        print("start train, batch_size={}(fake_batch={})".format(args.batch*args.fake_batch, args.fake_batch))
        while True:
            if hasattr(framework, 'get_lr'):
                lr = framework.get_lr(steps, batchSize)
            else:
                lr = get_lr(steps)
            t0 = default_timer()
            data_time = 0
            tflearn.is_training(True, session=sess)

            D_step = args.D_step if args.D_step is not None else args.pre_step//2
            if framework.discriminator:
                if steps<D_step:
                    summ, t1 = update_step('R', summopt=framework.summaryExtra, learningRate=lr)
                else:
                    if steps==D_step:
                        zerocnt = 0
                        print("pre train D for {} steps".format(args.pre_step))
                        for i in range(args.pre_step):
                            Dt0 = default_timer()
                            Dsumm, t1 = update_step('T', summopt=framework.summaryExtra, pos_learningRate=lr)
                            for v in tf.Summary().FromString(Dsumm).value:
                                if v.tag == 'Triplet_loss':
                                    Dloss = v.simple_value
                            if Dloss<=-10+1e-5:
                                zerocnt += 1
                                if zerocnt>=10: 
                                    print("early stop at D step:{}".format(i))
                                    break
                            if (i<500 and i%10==0) or i%500==0:
                                print('*%s* ' % run_id,
                                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                    'D train Steps %d, Total time %.2f, DLoss %.3e lr %.3e' % (i,
                                                                                                    default_timer() - Dt0,
                                                                                                    Dloss,
                                                                                                    lr),
                                    end='\n', flush=True)
                            
                    _, t1 = update_step('T', summopt=framework.summaryExtra, pos_learningRate=lr)
                    data_time += t1
                    summ, t1 = update_step('RD', summopt=framework.summaryExtra, learningRate=lr)
                data_time += t1
            else:
                summ, t1 = update_step('R', summopt=framework.summaryExtra, learningRate=lr)
                data_time += t1

            Dloss = -1
            for v in tf.Summary().FromString(summ).value:
                if (args.discriminator is None or steps<args.D_step) and v.tag == 'loss':
                    loss = v.simple_value
                if args.discriminator and steps>=args.D_step and v.tag == 'RD_loss':
                    loss = v.simple_value
                if v.tag == 'Triplet_loss':
                    Dloss = v.simple_value

            steps += 1
            if args.debug or steps % 10 == 0:
                if steps >= args.epochs * iterationSize:
                    break

                if not args.debug:
                    summaryWriter.add_summary(summ, steps)

                if steps % 1000 == 0:
                    if hasattr(framework, 'summaryImages'):
                        summ, = sess.run([framework.summaryImages],
                                         set_tf_keys(fd))
                        summaryWriter.add_summary(summ, steps)

                if steps < 500 or steps % 500 == 0 and not args.debug:
                    print('*%s* ' % run_id,
                          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                          'Steps %d, Total time %.2f, data %.2f%%. Loss %.3e lr %.3e DLoss %.3e' % (steps,
                                                                                         default_timer() - t0,
                                                                                         data_time/ (
                                                                                             default_timer() - t0),
                                                                                         loss,
                                                                                         lr,
                                                                                         Dloss),
                          end='\n', flush=True)

                if time.time() - last_save_stamp > 3600 or steps % iterationSize == iterationSize - 500:
                    last_save_stamp = time.time()
                    saver.save(sess, os.path.join(modelPrefix, 'model'),
                               global_step=steps, write_meta_graph=False)

                if args.debug or steps % args.val_steps == 0:
                    try:
                        t0 = default_timer()
                        val_gen = dataset.generator(
                            Split.VALID, loop=False, batch_size=batchSize)
                        metrics = framework.validate(
                            sess, val_gen, summary=True)
                        val_summ = tf.Summary(value=[
                            tf.Summary.Value(tag='val_' + k, simple_value=v) for k, v in metrics.items()
                        ])
                        summaryWriter.add_summary(val_summ, steps)
                        print('Step {}, validation dice {}, validation time {}'.format(steps, metrics['dice_score'], default_timer()-t0), flush=True)
                    except:
                        if steps == args.val_steps:
                            print('Step {}, validation failed!'.format(steps), flush=True)
    print('Finished.')


if __name__ == '__main__':
    main()
