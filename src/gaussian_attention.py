import tensorflow as tf
import numpy as np

def gaussian_mask(u, s, d, R, C, n_channels):
    """
        :param u: tf.Tensor of size (batch_size, 1), centre of the first Gaussian.
        :param s: tf.Tensor of size (batch_size, 1), standard deviation of Gaussians.
        :param d: tf.Tensor of size (batch_size, 1), shift between Gaussian centres.
        :param R: int, number of rows in the mask, there is one Gaussian per row.
        :param C: int, number of columns in the mask.
    """
    # indices to create centres
    # expand the tensors to allow batch sizes > 1
    d = tf.expand_dims(d, 1)
    R = tf.tile(tf.to_float(tf.reshape(tf.range(R), (1, 1, R))), [tf.shape(u)[0], 1, 1])
    C = tf.tile(tf.to_float(tf.reshape(tf.range(C), (1, C, 1))), [tf.shape(u)[0], 1, 1])

    # construct the gaussians
    centres = R * d + u[:, np.newaxis]
    column_centres = C - centres
    # we add eps for numerical stability
    mask = tf.exp(-.5 * tf.square(column_centres / tf.reshape(s, (-1, 1, 1))))
    
    normalized_mask = mask / (tf.reduce_sum(mask, 1, keepdims=True) + 1e-8)

    # repeat the 
    normalized_mask = tf.stack([normalized_mask]*n_channels, axis=-1)

    return normalized_mask

def gaussian_attention(img_tensor, transform_params, crop_size):
    """
        :param img_tensor: tf.Tensor of size (batch_size, Height, Width, channels)
        :param transform_params: tf.Tensor of size (batch_size, 6), where params are  (mean_y, std_y, d_y, mean_x, std_x, d_x) specified in pixels.
        :param crop_size: tuple of 2 ints, size of the resulting crop
    """
    # parse arguments
    h, w = crop_size
    H, W, n_channels = img_tensor.shape.as_list()[1:4]

    split_ax = transform_params.shape.ndims -1
    
    uy, sy, dy, ux, sx, dx = tf.split(transform_params, 6, split_ax)
    
    # create Gaussian masks, one for each axis
    Ay = gaussian_mask(uy, sy, dy, h, H, n_channels)
    Ax = gaussian_mask(ux, sx, dx, w, W, n_channels)

    # take care of the color channel
    Ay = tf.transpose(Ay, perm=[0,3,1,2])
    Ax = tf.transpose(Ax, perm=[0,3,1,2])
    img_tensor = tf.transpose(img_tensor, perm=[0,3,1,2])

    # extract glimpse
    glimpse = tf.matmul(tf.matmul(Ay, img_tensor, adjoint_a=True), Ax)
    
    # take care of the color channel
    glimpse = tf.transpose(glimpse, perm=[0,2,3,1])

    return glimpse