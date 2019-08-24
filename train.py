import numpy as np
import datetime
from network import CycleGan
import os
import keras.backend as ktf
from util import Util


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# ktf.set_session(tf.Session(config=config))

epochs = 100
batch_size = 4
img_height = 256
img_weight = 256
model_path = 'model'
dataset_name = 'monet'
dataset_path = 'datasets/monet2photo/'
output = "output"
util = Util(dataset_path=dataset_path, img_height=img_height, img_weight=img_weight)


def train(cycle, g_A2B, g_B2A, d_A, d_B, disc_patch, sample_interval=200):
    start_time = datetime.datetime.now()

    # Adversarial loss ground truths
    valid = np.ones((batch_size,) + disc_patch)
    fake = np.zeros((batch_size,) + disc_patch)

    g_loss_best = 9999.0

    for epoch in range(epochs):
        for batch_i, (imgs_A, imgs_B) in enumerate(util.load_batch(batch_size)):
            # transfer to another domain

            fake_B = g_A2B.predict(imgs_A)
            fake_A = g_B2A.predict(imgs_B)

            # Train the discriminators (original images = real / translated = Fake)
            dA_loss_real = d_A.train_on_batch(imgs_A, valid)
            dA_loss_fake = d_A.train_on_batch(fake_A, fake)
            dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

            dB_loss_real = d_B.train_on_batch(imgs_B, valid)
            dB_loss_fake = d_B.train_on_batch(fake_B, fake)
            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

            # Total disciminator loss
            d_loss = 0.5 * np.add(dA_loss, dB_loss)

            # Train the generators
            g_loss = cycle.train_on_batch([imgs_A, imgs_B],
                                             [valid, valid,
                                              imgs_A, imgs_B,
                                              imgs_A, imgs_B])

            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                % (epoch, epochs,
                   batch_i, data_loader.n_batches,
                   d_loss[0], 100 * d_loss[1],
                   g_loss[0],
                   np.mean(g_loss[1:3]),
                   np.mean(g_loss[3:5]),
                   np.mean(g_loss[5:6]),
                   elapsed_time))

            # If at save interval => save generated image samples
            if batch_i % sample_interval == 0:
                util.sample_images(g_AB, g_BA, epoch, batch_i, output, dataset_name)
                # save best gen
                if g_loss[0] < g_loss_best:
                    g_loss_best = g_loss[0]
                    g_AB.save(os.path.join(model_path, 'gen_a2b.h5'))
                    g_BA.save(os.path.join(model_path, 'gen_b2a.h5'))
    cycle.save(os.path.join(model_path, 'cycle_gan.h5'))


# Calculate output shape of D (PatchGAN)
patch_h = int(img_height / 2 ** 4)
patch_w = int(img_weight / 2 ** 4)
disc_patch = (patch_h, patch_w, 1)

cycle_gan = CycleGan(img_height, img_height)

# generator transfer domain a to domain b
gen_a2b = cycle_gan.build_generator()

# generator transfer domain b to domain a
gen_b2a = cycle_gan.build_generator()

# build_discriminator for domain a
dis_a = cycle_gan.build_discriminator()

# build_discriminator for domain b
dis_b = cycle_gan.build_discriminator()

# cycle gan
model = cycle_gan.cycle_gan(gen_a2b, gen_b2a, dis_a, dis_b)

# training
train(model, gen_a2b, gen_b2a, dis_a, dis_b, disc_patch)
