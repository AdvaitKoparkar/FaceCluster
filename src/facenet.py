# trained weights from https://github.com/R4j4n/Face-recognition-Using-Facenet-On-Tensorflow-2.X/tree/master
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, Input, MaxPooling2D, Dense, Dropout, BatchNormalization, Concatenate, GlobalAveragePooling2D

class ConvBlock(tf.keras.Model):
    def __init__(self, filters : int , kernel : int , strides : int , padding : str , use_bias : bool , name : str ) -> None :
        super(ConvBlock, self).__init__(name=name)
        self.conv = Conv2D(
            filters=filters, 
            kernel_size=kernel, 
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            name=f'{name}_Conv',
        )
        
        self.bn = BatchNormalization(
            axis=3, 
            momentum=0.995, 
            epsilon=0.001, 
            scale=False, 
            name=f'{name}_BatchNorm',
        )
    
        self.activation = Activation(
            'relu', 
            name=f'{name}_Activation',
        )

    def call(self, x : tf.Tensor ) -> tf.Tensor :
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class InceptionResnetBlock(tf.keras.Model):
    def __init__(self, branches : list, dimension : int , scale : float, name : str) -> None :
        super(InceptionResnetBlock, self).__init__(name=name)
        self.branches = []
        for branch_idx, branch in  enumerate(branches):
            self.branches.append(
                tf.keras.models.Sequential([
                    ConvBlock(filters, kernel, strides, padding, use_bias, name=f'{name}_Branch{branch_idx}_layer{layer_idx}')
                    for layer_idx, (filters, kernel, strides, padding, use_bias) in enumerate(branch)
                ], name=f'{name}_Branch{branch_idx}')
            )

        self.conv = Conv2D(dimension, 1, 1, 'same', use_bias=True, name=f'{name}_Conv2d_1x1')
        self.scale = scale
        self.activation = Activation('relu', name=f'{name}_Activation')

    def call(self, x):
        branches  = [
            branch(x) for branch in self.branches
        ]
        mixed = Concatenate(axis=3, name=f'{self.name}_Concatenate')(branches)
        up = self.conv(mixed)
        up = up * self.scale
        x  = x + up
        x  = self.activation(x)
        return x

class ReductionBlock(tf.keras.Model):
    def __init__(self, branches : int, pool : list, name : str):
        super(ReductionBlock, self).__init__(name=name)
        self.branches = []
        for branch_idx, branch in enumerate(branches):
            self.branches.append(
                tf.keras.models.Sequential([
                    ConvBlock(filters, kernel, strides, padding, use_bias, name=f'{name}_Branch{branch_idx}_layer{layer_idx}')
                    for layer_idx, (filters, kernel, strides, padding, use_bias) in enumerate(branch)
                ], name=f'{name}_Branch{branch_idx}')
            )

        if pool is not None:
            self.branches.append(
                MaxPooling2D(pool[0], pool[1], name=f'{name}_Pool_{pool[0]}x{pool[0]}')
            )
    
    def call(self, x):
        branches = [
            branch(x) for branch in self.branches
        ]
        x = Concatenate(axis=3, name=f'{self.name}_Concatenate')(branches)
        return x

def FacenetInception(path : str) -> tf.keras.Model :
    model = tf.keras.models.Sequential([
        Input(shape=(160, 160, 3)),
        ConvBlock(32, 3, 2, 'valid', False, 'Conv2d_1'),
        ConvBlock(32, 3, 1, 'valid', False, 'Conv2d_2'),
        ConvBlock(64, 3, 1, 'same',  False, 'Conv2b_3'),
        MaxPooling2D(3, 2, name='MaxPool_3'),

        ConvBlock(80,  1, 1, 'valid', False, 'Conv2d_4'),
        ConvBlock(192, 3, 1, 'valid', False, 'Conv2d_5'),
        ConvBlock(256, 3, 2, 'valid', False, 'Conv2d_6'),
        
        # InceptionBlockA
        tf.keras.Sequential([
            InceptionResnetBlock(
            branches=[
                # branch0
                [
                    [32, 1, 1, 'same', False],
                ],
                # branch1
                [
                    [32, 1, 1, 'same', False],
                    [32, 3, 1, 'same', False],
                ],
                # branch2
                [
                    [32, 1, 1, 'same', False],
                    [32, 3, 1, 'same', False],
                    [32, 3, 1, 'same', False],
                ],
            ], dimension=256, scale=0.17, name=f'InceptionBlockA-{idx}')
            for idx in range(5)
        ], name='InceptionBlockA'),

        # # ReductionBlockA
        ReductionBlock(
            branches=[
                # branch0
                [
                    [384, 3, 2, 'valid', False],
                ],
                # branch1
                [
                    [192, 1, 1, 'same', False],
                    [192, 3, 1, 'same', False],
                    [256, 3, 2, 'valid', False],
                ]
            ], pool = [3, 2], name='RecutionBlockA'
        ),

        # InceptionBlockB
        tf.keras.Sequential([
            InceptionResnetBlock(
            branches=[
                # branch0
                [
                    [128, 1, 1, 'same', False],
                ],
                # branch1
                [
                    [128, 1, 1, 'same', False],
                    [128, [1, 7], 1, 'same', False],
                    [128, [7, 1], 1, 'same', False],
                ],
            ], dimension=896, scale=0.1, name=f'InceptionBlockB-{idx}')
            for idx in range(10)
        ], name='InceptionBlockB'),

        # ReductionBlockB
        ReductionBlock(
            branches=[
                # branch0
                [
                    [256, 1, 1, 'same', False],
                    [384, 3, 2, 'valid', False],
                ],
                # branch1
                [
                    [256, 1, 1, 'same', False],
                    [256, 3, 2, 'valid', False],
                ],
                # branch2
                [
                    [256, 1, 1, 'same', False],
                    [256, 3, 1, 'same', False],
                    [256, 3, 2, 'valid', False],
                ],
            ], pool = [3, 2], name='RecutionBlockB'
        ),

        # # InceptionBlockC
        tf.keras.Sequential([
            InceptionResnetBlock(
            branches=[
                # branch0
                [
                    [192, 1, 1, 'same', False],
                ],
                # branch1
                [
                    [192, 1, 1, 'same', False],
                    [192, [1, 3], 1, 'same', False],
                    [192, [3, 1], 1, 'same', False],
                ],
            ], dimension=1792, scale=0.2, name=f'InceptionBlockC-{idx}')
            for idx in range(6)
        ], name='InceptionBlockC'),

        # classification
        GlobalAveragePooling2D(name='AvgPool'),
        Dropout(1.0 - 0.8, name='Dropout'),
        Dense(128, use_bias=False, name='Bottleneck'),
        BatchNormalization(momentum=0.995, epsilon=0.001, scale=False, name='Bottleneck_BatchNorm'),
    ], name='facenet')

    model.load_weights(path)
    return model

if __name__ == '__main__':
    import numpy as np
    model = FacenetInception('facenet_inception.h5')
    model.summary()
    random_tensor = np.random.randn(1, 160, 160, 3)
    print(model.predict(random_tensor).shape)
