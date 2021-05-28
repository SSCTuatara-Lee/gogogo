import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

import source.neuralnet as nn
import source.datamanager as dman
import source.tf_process as tfp
#import source.stamper as stamper
tf.compat.v1.disable_eager_execution()


def main():

    srnet = nn.SRNET()           #神经网络初始化

    dataset = dman.DataSet()     #调用图像的数据

    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()       #tensorflow 的全局变量初始化，训练模型初始化

    tfp.training(sess=sess, neuralnet=srnet, saver=saver, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch)
    tfp.validation(sess=sess, neuralnet=srnet, saver=saver, dataset=dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=5000, help='-')   #进行5000次迭代，每一次迭代都对图像数据进行损失函数和PSNR运算
    parser.add_argument('--batch', type=int, default=16, help='-')

    FLAGS, unparsed = parser.parse_known_args()

    main()
