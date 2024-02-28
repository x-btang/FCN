from tensorflow.keras import Model, layers
from tensorflow.keras.applications import vgg16
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Reshape, Concatenate, Conv2DTranspose


def UNet(nClasses, input_height, input_width):
    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    """Encoder"""
    vgg_streamlined = vgg16.VGG16(
        include_top=False,
        weights='imagenet',
        input_tensor=img_input
    )

    assert isinstance(vgg_streamlined, Model)

    """Decoder block 1"""
    x = vgg_streamlined.get_layer(name='block5_conv3').output
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='valid')(x)
    """skip connection"""
    x = Concatenate(axis=-1)([vgg_streamlined.get_layer(name='block4_conv3').output, x])
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    """Decoder block 2"""
    x = Conv2DTranspose(256, 2, 2, padding='valid')(x)
    """skip connection"""
    x = Concatenate(axis=-1)([vgg_streamlined.get_layer(name='block3_conv3').output, x])
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    """Decoder block 3"""
    x = Conv2DTranspose(128, 2, 2, padding='valid')(x)
    """skip connection"""
    x = Concatenate(axis=-1)([vgg_streamlined.get_layer(name='block2_conv2').output, x])
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    """Decoder block 4"""
    x = Conv2DTranspose(64, 2, 2, padding='valid')(x)
    """skip connection"""
    x = Concatenate(axis=-1)([vgg_streamlined.get_layer(name='block1_conv2').output, x])
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    """segmentation mask"""
    x = Conv2D(nClasses, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((-1, nClasses))(x)
    x = Activation('softmax')(x)
    model = Model(inputs=img_input, outputs=x)

    return model


if __name__ == '__main__':
    unet = UNet(15, 320, 320)
    """print(unet.get_weights()[2]), 查看权重是否改变, 加载vgg权重测试使用"""
    from keras.utils import plot_model
    plot_model(unet, show_shapes=True, to_file='unet_model.png')
    print(len(unet.layers))
    unet.summary()
