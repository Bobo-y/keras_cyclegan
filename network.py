from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16


class CycleGan:

    def __init__(self, height, weight, channels=3):
        self.height = height
        self.weight = weight
        self.channels = channels
        self.img_shape = (self.height, self.weight, self.channels)

    def build_generator(self):
        # U-net like based on vgg16
        input_img = Input(name='input_img',
                          shape=(self.height,
                                 self.weight,
                                 self.channels),
                          dtype='float32')
        vgg16 = VGG16(input_tensor=input_img,
                      weights='imagenet',
                      include_top=False)
        vgg_pools = [vgg16.get_layer('block%d_pool' % i).output
                     for i in range(1, 6)]

        def decoder(layer_input, skip_input, channel, last_block=False):
            if not last_block:
                concat = Concatenate(axis=-1)([layer_input, skip_input])
                bn1 = InstanceNormalization()(concat)
            else:
                bn1 = InstanceNormalization()(layer_input)
            conv_1 = Conv2D(channel, 1,
                            activation='relu', padding='same')(bn1)
            bn2 = InstanceNormalization()(conv_1)
            conv_2 = Conv2D(channel, 3,
                            activation='relu', padding='same')(bn2)
            return conv_2

        d1 = decoder(UpSampling2D((2, 2))(vgg_pools[4]), vgg_pools[3], 256)
        d2 = decoder(UpSampling2D((2, 2))(d1), vgg_pools[2], 128)
        d3 = decoder(UpSampling2D((2, 2))(d2), vgg_pools[1], 64)
        d4 = decoder(UpSampling2D((2, 2))(d3), vgg_pools[0], 32)
        d5 = decoder(UpSampling2D((2, 2))(d4), None, 32, True)

        output = Conv2D(3, 3, activation='tanh', padding='same')(d5)
        model = Model(inputs=input_img, outputs=output)
        # model.summary()
        return model

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        image = Input(shape=self.img_shape)

        d1 = d_layer(image, 64, normalization=False)
        d2 = d_layer(d1, 128)
        d3 = d_layer(d2, 256)
        d4 = d_layer(d3, 512)

        patch_out = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        discriminator = Model(image, patch_out)
        optimizer = Adam(0.0002, 0.5)
        discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        # discriminator.summary()
        return discriminator

    def cycle_gan(self, gen_a2b, gen_b2a, dis_a, dis_b):
        image_a = Input(shape=self.img_shape)
        image_b = Input(shape=self.img_shape)

        fake_b = gen_a2b(image_a)
        fake_a = gen_b2a(image_b)

        reconstr_a = gen_b2a(fake_b)
        reconstr_b = gen_a2b(fake_a)

        img_a_identity = gen_b2a(image_a)
        img_b_identity = gen_a2b(image_b)

        dis_a.trainable = False
        dis_b.trainable = False

        patch_out_a = dis_a(fake_a)
        patch_out_b = dis_b(fake_b)

        cycle_model = Model(inputs=[image_a, image_b],
                            outputs=[patch_out_a, patch_out_b,
                                     reconstr_a, reconstr_b,
                                     img_a_identity, img_b_identity])
        optimizer = Adam(0.0002, 0.5)
        lambda_cycle = 10.0  # Cycle-consistency loss
        lambda_id = 0.1 * lambda_cycle  # Identity loss
        cycle_model.compile(loss=['mse', 'mse',
                                  'mae', 'mae',
                                  'mae', 'mae'],
                            loss_weights=[1, 1,
                                          lambda_cycle, lambda_cycle,
                                          lambda_id, lambda_id],
                            optimizer=optimizer)
        # cycle_model.summary()
        return cycle_model


if __name__ == "__main__":
    cycle_gan = CycleGan(256, 256)
    # generator transfer domain a to domain b
    gen_a2b = cycle_gan.build_generator()
    # generator transfer domain b to domain a
    # gen_b2a = cycle_gan.build_generator()
    # build_discriminator for domain a
    dis_a = cycle_gan.build_discriminator()
    # build_discriminator for domain b
    dis_b = cycle_gan.build_discriminator()

    model = cycle_gan.cycle_gan(gen_a2b, gen_b2a, dis_a, dis_b)

