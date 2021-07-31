import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import random
from tensorflow.python.keras.losses import LossFunctionWrapper
import sys
import time

# Code based on Alhashim et al.
# Original code can be found at https://github.com/ialhashim/DenseDepth/blob/master/loss.py

# This script contains all loss function components + the ordinal error metric function

# SI log loss as introduced by Eigen et al. (2014)
def depth_loss_function(y_true, y_pred):
    a = 10
    l = 0.85

    #new BTS style version:
    err = K.log(K.exp(y_pred)) - K.log(y_true)
    silog = K.sqrt(K.mean(K.pow(err, 2)) - l * K.mean(K.pow(err, 2))) * a

    return silog

# Alternative version provided by https://github.com/julilien (not used in current implementation anymore)
class SIError(LossFunctionWrapper):
    """
    As described by Eigen et al., 2014
    """
    def __init__(self, lambda_val=0.5, loss_fn=None, name="si_error"):
        if loss_fn is None:
            loss_fn = self.loss
        super(SIError, self).__init__(fn=loss_fn, name=name)
        self.lambda_val=lambda_val

    def loss(self, y_true, y_pred):
        if tf.rank(y_true) == 2:
            y_true = tf.expand_dims(y_true, axis=0)
        if tf.rank(y_pred) == 2:
            y_pred = tf.expand_dims(y_pred, axis=0)

        y_true = K.clip(y_true, 1e-7, None)

        y_true_log = tf.math.log(y_true)

        log_diff = y_pred - y_true_log

        log_diff_square = tf.math.square(log_diff)
        sum1 = tf.reduce_mean(log_diff_square, axis=[1, 2])
        sum2 = tf.math.square(tf.math.reduce_mean(log_diff, axis=[1, 2]))

        result = sum1 - self.lambda_val * sum2

        return result

class ScaledSIError(SIError):
    def __init__(self, lambda_val=0.85, alpha_val=10.):
        #Default parameters as used in BTS
        super(ScaledSIError, self).__init__(lambda_val=lambda_val, loss_fn=self.loss, name="scaled_si_error")
        self.lambda_val = lambda_val
        self.alpha_val = alpha_val

    def loss(self, y_true, y_pred):
        return self.alpha_val * tf.math.sqrt(super(ScaledSIError, self).loss(y_true, y_pred))

# Ordinal error metric implementation (currently inefficient and only works as metric outside of training)
def ranking_loss_function_ordinal_err(y_true, y_pred):

    if y_true.shape[0] != None:
        num_samples = 500
        total_samples = num_samples ** 2

        population = [(x, y) for x in range(y_true.shape[1]) for y in range(y_true.shape[2])]

        sample_points = random.sample(population, num_samples)

        ord_err_list = [0 for x in range(y_true.shape[0])]

        for i in range(y_true.shape[0]):
            for x1, y1 in sample_points:
                for x2, y2 in sample_points:
                    if y_true[i][x1][y1] > y_true[i][x2][y2]:
                        image_relation_true = 1
                    elif y_true[i][x1][y1] < y_true[i][x2][y2]:
                        image_relation_true = -1
                    else:
                        image_relation_true = 0

                    if y_pred[i][x1][y1] > y_pred[i][x2][y2]:
                        image_relation_pred = 1
                    elif y_pred[i][x1][y1] < y_pred[i][x2][y2]:
                        image_relation_pred = -1
                    else:
                        image_relation_pred = 0

                    if image_relation_true != image_relation_pred:
                        ord_err_list[i] = ord_err_list[i] + 1

        ord_err_list = np.divide(ord_err_list, total_samples)

        return np.mean(ord_err_list)

    else:
        # Safety check; was not executed locally yet
        tf.print("[DEBUG] y_true shape is 0", output_stream=sys.stdout)
        return None

# Ranking loss function as explained in the thesis
def ranking_loss_function(y_true, y_pred):
    # Based on In the Wild ranking loss (by Chen et al. (2016))

    # hard-coded here just to make sure as the same shapes were used for all performed trainings
    y_true_shape = (4, 480, 640, 1)
    y_pred_shape = (4, 480, 640, 1)

    if y_true_shape[0] != None:
        num_sample_pairs = 2500

        bs = y_true_shape[0]
        total_samples = num_sample_pairs * bs

        sample_pairs = create_random_samples(y_true, y_pred, num_sample_pairs)

        # flatten depth maps to gather points from 'one-dimensional image'
        y_true = tf.reshape(y_true, [bs, -1])
        y_pred = tf.reshape(y_pred, [bs, -1])

        # gather depth values to compute underlying relations after
        y_true_sel = tf.gather(y_true, tf.cast(sample_pairs, dtype=tf.int32), axis=1, batch_dims=1)
        y_pred_sel = tf.gather(y_pred, tf.cast(sample_pairs, dtype=tf.int32), axis=1, batch_dims=1)

        # compute depth relations
        relations = get_depth_relations(y_true_sel)

        # used to 'determine' whether point depths are equal or unequal
        mask = tf.abs(relations)

        # Compute loss value
        sample_loss = mask * tf.math.log(1 + tf.math.exp(-relations * (y_pred_sel[:, :, 0] - y_pred_sel[:, :, 1]))) + (1-mask) * tf.pow(y_pred_sel[:, :, 0] - y_pred_sel[:, :, 1], 2)

        sample_loss = tf.reduce_mean(sample_loss, axis=-1) # Originally only sum, but normalized here to increase metric performance

        sample_loss = tf.reduce_mean(sample_loss, axis=-1) # Compute mean over images in batch

        return sample_loss
    else:
        return None

# Random Sample Pair Generator
def create_random_samples(y_true, y_pred, num_pairs=10):

    # hard-coded currently just to make sure as all trainings were performed with exactly
    y_true_shape = (4, 480, 640, 1)
    y_pred_shape = (4, 480, 640, 1)
    total_num = num_pairs * y_true_shape[0]

    if y_true_shape[0] != None:

        # create random point locations in flattened image
        sample_pairs = np.random.randint(0, y_true_shape[1] * y_true_shape[2] + 1, (total_num * 2))

        # reshape to point pairs
        sample_pairs = np.reshape(sample_pairs, (y_true_shape[0], num_pairs, 2))

        return sample_pairs

# Determine depth relations between pairs element-wise
# Based on https://github.com/julilien/PLDepth
def get_depth_relations(depth_values, dtype=tf.float32):
    relation = tf.where(
        tf.greater_equal(
            (depth_values[:, :, 0]) / (depth_values[:, :, 1]),
            1.),
        tf.constant(1, dtype),
        tf.where(tf.greater(1.,
                            (depth_values[:, :, 0]) / (
                                    depth_values[:, :, 1])),
                 tf.constant(-1, dtype),
                 tf.constant(0, dtype)))

    return tf.cast(relation, dtype)

# Adjusted version of RMSE to take exp(y_pred) into consideration    
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(K.exp(y_pred) - y_true)))
