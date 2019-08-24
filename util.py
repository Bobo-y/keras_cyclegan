from glob import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image


class Util:
    def __init__(self, dataset_path, img_height, img_weight):
        self.dataset_path = dataset_path
        self.img_height = img_height
        self.img_weight = img_weight

    def load_data(self, domain, batch_size=1, is_testing=False):
        if is_testing:
            data_type = "test" + domain
        else:
            data_type = "train" + domain

        path = os.listdir(os.path.join(self.dataset_path, data_type))
        batch_images = np.random.choice(path, size=batch_size)
        imgs = []
        for img_path in batch_images:
            img = self.imread(os.path.join(self.dataset_path, data_type, img_path))
            if not is_testing:
                img = img.resize(img, (self.img_weight, self.img_height))
            else:
                img = img.resize(img, (self.img_weight, self.img_height))
            imgs.append(img)

        imgs = np.array(imgs)/127.5 - 1.
        return imgs

    def load_batch(self, batch_size=1, is_testing=False):

        if is_testing:
            data_type = "test"
        else:
            data_type = "train"

        path_A = glob(os.path.join(self.dataset_path, data_type + 'A/*'))

        path_B = glob(os.path.join(self.dataset_path, data_type + 'B/*'))

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = img_A.resize(img_A, (self.img_weight, self.img_height))
                img_B = img_B.resize(img_B, (self.img_weight, self.img_height))

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def imread(self, path):
        return np.array(Image.open(path)).astype(np.float)

    def sample_images(self, g_AB, g_BA, epoch, batch_i, output, dataset_name):
        r, c = 2, 3

        imgs_A = self.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.load_data(domain="B", batch_size=1, is_testing=True)

        # Translate images to the other domain
        fake_B = g_AB.predict(imgs_A)
        fake_A = g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = g_BA.predict(fake_B)
        reconstr_B = g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(output, dataset_name, str(epoch) + str(batch_i) + '.png'))
        plt.close()
