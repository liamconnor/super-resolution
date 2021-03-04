import numpy as np
import tensorflow as tf


DIV2K_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255
DIV2K_RGB_MEAN16 = np.array([0.4488, 0.4371, 0.4040]) * 2**16

def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]

def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch

def resolve_single16(model, lr):
    return resolve16(model, tf.expand_dims(lr, axis=0))[0]

def resolve16(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 2**16-1)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint16)
    return sr_batch

def evaluate(model, dataset):
    psnr_values = []
    for lr, hr in dataset:
        sr = resolve(model, lr)
        psnr_value = psnr(hr, sr)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)


# ---------------------------------------
#  Normalization
# ---------------------------------------

def normalize16(data, rgb_mean=DIV2K_RGB_MEAN16):
#    data = (data - data.min())/2.0**15
    data *= np.ones([1,1,1])
    data = data - tf.reduce_min(data)
    data *= DIV2K_RGB_MEAN16
    data = data * 2**15
    return data

#def normalize16(x, rgb_mean=DIV2K_RGB_MEAN16):
#    print(x)
#    print((x - rgb_mean) / 2**15)
#    return (x - rgb_mean) / 2**15

def denormalize16(data, rgb_mean=DIV2K_RGB_MEAN16):
    tf.print(tf.reduce_max(data))
    data = data*2**15
    data += rgb_mean
    tf.print(tf.reduce_max(data))
    return data

def normalize(x, rgb_mean=DIV2K_RGB_MEAN):
#    x = tf.math.subtract(x,tf.reduce_min(x))
#    printx * 127.5 + rgb_mean, 'norm2')    
    return (x - rgb_mean) / 127.5

def denormalize(x, rgb_mean=DIV2K_RGB_MEAN):
    return x * 127.5 + rgb_mean


def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0


def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5


# ---------------------------------------
#  Metrics
# ---------------------------------------


def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)


# ---------------------------------------
#  See https://arxiv.org/abs/1609.05158
# ---------------------------------------


def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


