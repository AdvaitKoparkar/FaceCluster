# TODO implement MTCNN module using https://github.com/xiangrufan/keras-mtcnn
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, PReLU, MaxPool2D, Permute, Flatten, Dense

class MTCNN(object):
    def __init__(self, config):
        self.config = config
        self._load_PNet()
        self._load_ONet()
        self._load_RNet()

    def _load_ONet(self):
        input = Input(shape=[48,48,3])
        x = Conv2D(32, (3,3), strides=1, padding='valid', name='conv1')(input)
        x = PReLU(shared_axes=[1,2], name='prelu1')(x)
        x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        x = Conv2D(64, (3,3), strides=1, padding='valid', name='conv2')(x)
        x = PReLU(shared_axes=[1,2], name='prelu2')(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        x = Conv2D(64, (3,3), strides=1, padding='valid', name='conv3')(x)
        x = PReLU(shared_axes=[1,2], name='prelu3')(x)
        x = MaxPool2D(pool_size=2)(x)

        x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
        x = PReLU(shared_axes=[1,2],name='prelu4')(x)
        x = Permute((3,2,1))(x)
        x = Flatten()(x)

        x = Dense(256, name='conv5') (x)
        x = PReLU(name='prelu5')(x)

        classifier = Dense(2, activation='softmax',name='conv6-1')(x)
        bbox_regress = Dense(4,name='conv6-2')(x)
        landmark_regress = Dense(10,name='conv6-3')(x)

        self.ONet = tf.keras.models.Model([input], [classifier, bbox_regress, landmark_regress])
        self.ONet.load_weights(self.config['ONet_weights'])

    def _load_PNet(self):
        input = Input(shape=[None, None, 3])
        x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
        x = PReLU(shared_axes=[1,2],name='PReLU1')(x)
        x = MaxPool2D(pool_size=2)(x)

        x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
        x = PReLU(shared_axes=[1,2],name='PReLU2')(x)

        x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
        x = PReLU(shared_axes=[1,2],name='PReLU3')(x)

        classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
        bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)
        self.PNet = tf.keras.models.Model([input], [classifier, bbox_regress])
        self.PNet.load_weights(self.config['PNet_weights'], by_name=True)

    def _load_RNet(self):
        input = Input(shape=[None, None, 3])
        x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
        x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
        x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)

        x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
        x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)

        x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
        x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
        x = Permute((3, 2, 1))(x)
        x = Flatten()(x)

        x = Dense(128, name='conv4')(x)
        x = PReLU( name='prelu4')(x)

        classifier = Dense(2, activation='softmax', name='conv5-1')(x)
        bbox_regress = Dense(4, name='conv5-2')(x)
        self.RNet = Model([input], [classifier, bbox_regress])
        self.RNet.load_weights(self.config['RNet_weights'], by_name=True)
