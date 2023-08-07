import argparse
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.dataflow import dataset
import tensorpack.dataflow.imgaug.deform
from tensorpack.tfutils.varreplace import remap_variables
import tensorflow as tf
from dorefa import get_dorefa
from tensorpack.train import SimpleTrainer, launch_train_with_config

BITW = 1
BITA = 2
BITG = 4
IMG_WIDTH = 64
class Model(ModelDesc):
    def inputs(self):
        # return [tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_WIDTH, 3], 'input'), tf.placeholder(tf.int32, [None], 'label')]
        return [tf.TensorSpec([None, IMG_WIDTH, IMG_WIDTH, 3], tf.float32, 'input'), tf.TensorSpec([None], tf.int32, 'label')]
    def build_graph(self, image, label):
        is_training = get_current_tower_context().is_training
        fw, fa, fg = get_dorefa(BITW, BITA, BITG)
        # monkey-patch tf.get_variable to apply fw
        def binarize_weight(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or 'conv0' in name or 'fc' in name:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)
        def cabs(x):
            return tf.clip_by_value(x, 0.0, 1.0, name='cabs')
        def activate(x):
            return fa(cabs(x))
        
        image = image / 256.0
        
        with remap_variables(binarize_weight), argscope(BatchNorm, momentum=0.9, epsilon=1e-4), argscope(Conv2D, use_bias=False):
            logits = (LinearWrap(image)
            .Conv2D('conv0', 64, 5, padding='SAME', use_bias=True)
            .MaxPooling('pool0', 2, padding='SAME')
            .apply(activate)
            # 32
            .Conv2D('conv1', 128, 3, padding='SAME')
            .apply(fg)
            .BatchNorm('bn1').apply(activate)
            # 32
            .Conv2D('conv2', 128, 3, padding='SAME')
            .apply(fg)
            .BatchNorm('bn2')
            .MaxPooling('pool1', 2, padding='SAME')
            .apply(activate)
            # 16
            .Conv2D('conv3', 128, 3, padding='SAME')
            .apply(fg)
            .BatchNorm('bn3').apply(activate)
            # 16
            .Conv2D('conv4', 128, 3, padding='VALID')
            .apply(fg)
            .BatchNorm('bn4').apply(activate)
            .MaxPooling('pool2', 2, padding='SAME')
            # 7
            .Conv2D('conv5',256, 3, padding='VALID')
            .apply(fg)
            .BatchNorm('bn5').apply(activate)
            # 5
            .tf.nn.dropout(0.5 if is_training else 1.0)
            .Conv2D('conv6', 256, 5, padding='VALID')
            .apply(fg).BatchNorm('bn6')
            .apply(activate)
            .FullyConnected('fc1', 10)())
        tf.nn.softmax(logits, name='output')
        # compute the number of failed samples
        wrong = tf.cast(tf.logical_not(tf.nn.in_top_k(logits, label, 1)), tf.float32, name='wrong_tensor')
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))
        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(1e-7))
        add_param_summary(('.*/W', ['histogram', 'rms']))
        total_cost = tf.add_n([cost, wd_cost], name='cost')
        add_moving_summary(cost, wd_cost, total_cost)
        return total_cost
        
    def optimizer(self):
        lr = tf.train.exponential_decay(
        learning_rate=1e-3,
        global_step=get_global_step_var(),
        decay_steps=4721 * 100,
        decay_rate=0.5, staircase=True, name='learning_rate')
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr, epsilon=1e-5)
    
def get_config():
    logger.auto_set_dir()
    # prepare dataset
    d1 = dataset.Cifar10('train')
    #d2 = dataset.Cifar10('extra')
    data_train = RandomMixData([d1])
    data_test = dataset.Cifar10('test')
    augmentors = [
        imgaug.Resize((IMG_WIDTH, IMG_WIDTH)),
        imgaug.Brightness(30),
        imgaug.Contrast((0.5, 1.5)),
        tensorpack.dataflow.imgaug.deform.GaussianDeform( [(0.2, 0.2), (0.2, 0.8), (0.8,0.8),
        (0.8,0.2)], (IMG_WIDTH,IMG_WIDTH), 0.2, 3),
    ]
    data_train = AugmentImageComponent(data_train, augmentors)
    data_train = BatchData(data_train, 128)
    data_train = PrefetchDataZMQ(data_train, 5)
    augmentors = [imgaug.Resize((IMG_WIDTH, IMG_WIDTH))]
    data_test = AugmentImageComponent(data_test, augmentors)
    data_test = BatchData(data_test, 128, remainder=True)
    #import sys
    #sys.exit(0)
    return TrainConfig(
        data=QueueInput(data_train),
        callbacks=[
        ModelSaver(),
        InferenceRunner(data_test,
        [ScalarStats('cost'), ClassificationError('wrong_tensor')])
        ],
        model=Model(),
        max_epoch=200,
    )

import cv2
import tensorflow as tf
import argparse
import numpy as np
import os
import sys
import scipy.io
import tensorpack.dataflow.imgaug.deform

def run_image(model, sess_init, inputs):
    pred_config = PredictConfig(
        model=model,
        session_init=sess_init,
        input_names=['input'],
        output_names=['output','IdentityN_18'] #IdentityN conv0/output pool0/output conv2/output gpuconv2 # IdentityN_3 bn1/output cabs conv0/W bn3/output # IdentityN_12 bn4
    )
    predictor = OfflinePredictor(pred_config)
    transformers = imgaug.AugmentorList([
        imgaug.Resize((IMG_WIDTH, IMG_WIDTH)),
        #imgaug.Brightness(30),
        #imgaug.Contrast((0.5, 1.5)),
        #tensorpack.dataflow.imgaug.deform.GaussianDeform( [(0.2, 0.2), (0.2, 0.8), (0.8,0.8), (0.8,0.2)], (64,64), 0.2, 3),
    ])
    correct = 0.
    idx = 0.
    classes = { 1: 'airplane',2:'automobile',3:'bird',4:'cat',5:'deer',6:'dog',7:'frog',8:'horse',9:'ship',10:'truck'}
    for f in inputs:
        assert os.path.isfile(f), f
        img = cv2.imread(f).astype('float32')
        #scipy.io.savemat('imgcv2',{'imgcv2':img})
        #sys.exit(0)
        assert img is not None
        img = transformers.augment(img)
        img = (img)[np.newaxis, :, :, :]
        outputs = predictor(img)
        #np.save('gpuconv1',outputs[1])
        scipy.io.savemat('gpuid6z',{'gpuid6z':outputs[1]})
        outputs = outputs[0]
        prob = outputs[0]
        ps = np.exp(prob) / np.sum(np.exp(prob), axis=0)
        predict= np.argmax(prob)
        #print('Predicted Class: {0}'.format(preds))
        curr_class = f.split('_')[1].split('.')[0]
        if curr_class == classes[predict+1]:
            correct += 1
            idx+=1
            print('==== Curr Class {0} , Index {1}, Correct {2}'.format(curr_class,idx,correct))
            print('#{2} Class {0} , Predicted {1}, Accuracy {3:.2f}'.format(curr_class,classes[predict+1],idx,correct/idx*100))
        ret = prob.argsort()[-10:][::-1]
        print(ret)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dorefa',
    help='number of bits for W,A,G, separated by comma. Defaults to \'1,2,4\'',
    default='1,2,4')
    parser.add_argument('--run', help='run on a list of images with the pretrained model',
    nargs='*')
    correct = 0
    args = parser.parse_args()
    if args.run:
        run_image(Model(), SaverRestore('train_log/cfr/checkpoint'), args.run)
        sys.exit()
    BITW, BITA, BITG = map(int, args.dorefa.split(','))
    config = get_config()
    #config.session_init = SaverRestore('train_log/cfr/checkpoint')
    print('BITW {0}, BITA {1}, BITG {2}'.format(BITW,BITA,BITG))
    launch_train_with_config(config, SimpleTrainer())