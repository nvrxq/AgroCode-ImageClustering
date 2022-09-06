import tensorflow as tf
import numpy as np

class CFG:
    '''
    Suppor Class
    '''

    @staticmethod
    def resize_images(images):
        return tf.image.resize(images, (512, 512), preserve_aspect_ratio=True)

    @staticmethod
    def _l2(tensor):
        return tf.math.l2_normalize(tensor, axis=1)

    @staticmethod
    def cosine_distance(a, b):
        return a.dot(b.T)




