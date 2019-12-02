# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""All user ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import gen_user_ops as _gen_user_ops
from tensorflow.python.framework import ops as _grad_helper
import tensorflow as _tf

from .gen_user_ops import correlation
from .gen_user_ops import reconstruction
from .gen_user_ops import covariance_internal as statistics
from .gen_user_ops import fast_leaky_relu
from .gen_user_ops import l1_matrix
from .gen_user_ops import reduce_sum_two_dim as reduce_sum_2d
from .gen_user_ops import affine_flow

def linear_similarity(X, Y):
    _, _, _, _, varXu, _, covXYu, _, _, cov, R = statistics(X, Y)
    beta = covXYu / (varXu + 1e-12)
    return cov, R, beta

@_grad_helper.RegisterGradient("Correlation")
def _correlation_grad(op, grad):
    """The gradients for `Correlation`.
    
    Args:
        op: The `Correlation` `Operation` to differentiate.
        grad: Gradient w.r.t. the output of the `Correlation` op.

    Returns:
        Gradients w.r.t. the input of the `Correlation` op.
    """
    dimg1, dimg2 = _gen_user_ops.correlation_gradient(
        img1=op.inputs[0],
        img2=op.inputs[1],
        dcorr=grad,
        kernel=op.get_attr('kernel'),
        displacement=op.get_attr('displacement'),
        stride=op.get_attr('stride'),
        normalise=op.get_attr('normalise'))
    return [dimg1, dimg2]

@_grad_helper.RegisterGradient("Reconstruction")
def _reconstruction_grad(op, grad):
    """The gradients for `Reconstruction`.
    
    Args:
        op: The `Reconstruction` `Operation` to differentiate.
        grad: Gradient w.r.t. the output of the `Reconstruction` op.

    Returns:
        Gradients w.r.t. the inputs of the `Reconstruction` op.
    """
    dimg2 = _gen_user_ops.reconstruction_wrt_image_gradient(
        img2=op.inputs[0],
        flow=op.inputs[1],
        dimg1=grad,
        coef1=op.get_attr('coef1'),
        coef2=op.get_attr('coef2'),
        coef3=op.get_attr('coef3'))
    dflow = _gen_user_ops.reconstruction_gradient(
        img2=op.inputs[0],
        flow=op.inputs[1],
        dimg1=grad,
        coef1=op.get_attr('coef1'),
        coef2=op.get_attr('coef2'),
        coef3=op.get_attr('coef3'))
    return [dimg2, dflow]

@_grad_helper.RegisterGradient("CovarianceInternal")
def _covariance_internal_grad(op, *grads):
    """The gradients for `CovarianceInternal`.
    
    Args:
        op: The `CovarianceInternal` `Operation` to differentiate.
        grads: Gradients w.r.t. the output of the `CovarianceInternal` op.

    Returns:
        Gradients w.r.t. the input of the `CovarianceInternal` op.
    """
    x = op.inputs[0]
    y = op.inputs[1]
    x_mean = op.outputs[2]
    y_mean = op.outputs[3]
    x_var_unscaled = op.outputs[4]
    y_var_unscaled = op.outputs[5]
    xy_pearson = op.outputs[10]
    dx, dy = _gen_user_ops.covariance_internal_gradient(
        x, y, x_mean, y_mean,
        x_var_unscaled, y_var_unscaled,
        xy_pearson,
        *grads,
        epsilon=op.get_attr('epsilon'))
    return [dx, dy]

@_grad_helper.RegisterGradient("FastLeakyRelu")
def _fast_leaky_relu_grad(op, grad):
    """The gradients for `FastLeakyRelu`.

    Args:
        op: The `FastLeakyRelu` `Operation` to differentiate.
        grad: Gradient w.r.t. the output of the `FastLeakyRelu` op.

    Returns:
        Gradients w.r.t. the input of the `FastLeakyRelu` op.
    """
    return _gen_user_ops.fast_leaky_relu_gradient(op.inputs[0],
        grad,
        leaking=op.get_attr('leaking'))

@_grad_helper.RegisterGradient("L1Matrix")
def _l1_matrix_grad(op, grad):
    """The gradients for `L1Matrix`.

    Args:
        op: The `L1Matrix` `Operation` to differentiate.
        grad: Gradient w.r.t. the output of the `L1Matrix` op.

    Returns:
        Gradients w.r.t. the input of the `L1Matrix` op.
    """
    din1, din2 = _gen_user_ops.l1_matrix_gradient(
        op.inputs[0],
        op.inputs[1],
        grad,
        amplifier=op.get_attr('amplifier'))
    return [din1, din2]

@_grad_helper.RegisterGradient("ReduceSumTwoDim")
def _reduce_sum_2d_grad(op, grad):
    """The gradients for `ReduceSumTwoDim`.

    Args:
        op: The `ReduceSumTwoDim` `Operation` to differentiate.
        grad: Gradient w.r.t. the output of the `ReduceSumTwoDim` op.

    Returns:
        Gradients w.r.t. the input of the `ReduceSumTwoDim` op.
    """
    in_length = _tf.shape(op.inputs[0])[1]
    grad = _tf.tile(_tf.reshape(grad, [-1, 1]), _tf.stack([1, in_length]))
    return grad

def _entropy_from_l1_matrix(l1_mat, s1, s2, k2):
    """Internal method, computes unscaled entropy from L1 matrix.

    Args:
        l1_mat: of shape [batch, s1, s2]
        s1, s2: integers, number of samples
        k2: coefficient to normalize probability
    
    Returns:
        unscaled entropies of shape [batch]
    """
    delta_mat = _tf.sigmoid(l1_mat)
    count_mat = reduce_sum_2d(_tf.reshape(delta_mat, [-1, s2]))
    logpr_mat = _tf.log(k2 * count_mat)
    entropies = reduce_sum_2d(_tf.reshape(logpr_mat, [-1, s1]))
    return entropies

def unscaled_entropies(img1, img2, bin_width, sample=2048, seed=None):
    """Computes stochastic continuated entropies (unscaled) from two images.

    Args:
        img1, img2: Two `Tensor`s of the same floating-point type.
            The two arguments should have the same length for the
            first four dimensions.
        bin_width: A floating-point number (cannot be `Tensor`).
            The width of a bin.
        sample: Optional, number of samples, defaults to 2048.
        seed: Optional, random seed for samples, defaults to None.

    Returns:
        Three `Tensor`s of shape [batch]: H(img1), H(img2), H(img1, img2).
        The entropies are not scaled. To obtain the usual entropies,
        multiply them by -1/sample.
    """
    img_shape = _tf.shape(img1)
    batch = img_shape[0]
    size = img_shape[1] * img_shape[2] * img_shape[3]
    channels1 = img_shape[4]
    channels2 = _tf.shape(img2)[4]
    new_shape1 = _tf.stack([batch, size * channels1])
    new_shape2 = _tf.stack([batch, size * channels2])
    img1 = _tf.reshape(img1, new_shape1)
    img2 = _tf.reshape(img2, new_shape2)
    samples = _tf.random_uniform([sample], 0, size, _tf.int32, seed)
    sample_shape = _tf.stack([batch, sample, 1])
    img1s = _tf.reshape(_tf.gather(img1, samples * channels1, axis=-1),
        sample_shape)
    img2s = _tf.reshape(_tf.gather(img2, samples * channels2, axis=-1),
        sample_shape)
    amp = -2 / bin_width
    obs_factor = 2 / sample
    img1_l1 = l1_matrix(img1s, img1s, amplifier=amp)
    img2_l1 = l1_matrix(img2s, img2s, amplifier=amp)
    h1 = _entropy_from_l1_matrix(img1_l1, sample, sample, obs_factor)
    h2 = _entropy_from_l1_matrix(img2_l1, sample, sample, obs_factor)
    h12 = _entropy_from_l1_matrix(_tf.minimum(img1_l1, img2_l1),
        sample, sample, obs_factor)
    return h1, h2, h12

def normalised_mutual_info(img1, img2, bin_width, sample=2048, seed=None):
    """Computes stochastic continuated NMI between two images.

    Args:
        img1, img2: Two `Tensor`s of the same floating-point type.
            The two arguments should have the same length for the
            first four dimensions.
        bin_width: A floating-point number (cannot be `Tensor`).
            The width of a bin.
        sample: Optional, number of samples, defaults to 2048.
        seed: Optional, random seed for samples, defaults to None.

    Returns:
        A `Tensor` of shape [batch], NMI.
    """
    h1, h2, h12 = unscaled_entropies(img1, img2, bin_width, sample, seed)
    return (h1 + h2) / h12

@_grad_helper.RegisterGradient("AffineFlow")
def _affine_flow_grad(op, grad):
    """The gradients for `AffineFlow`.
    
    Args:
        op: The `AffineFlow` `Operation` to differentiate.
        grad: Gradient w.r.t. the output of the `AffineFlow` op.

    Returns:
        Gradients w.r.t. the inputs of the `AffineFlow` op.
    """
    dW, db = _gen_user_ops.affine_flow_gradient(grad,
        op.get_attr("len1"),
        op.get_attr("len2"),
        op.get_attr("len3"))
    return [dW, db]

def det3x3(M):
    """Computes the determinant of 3x3 matrices.

    Args:
        M: A `Tensor` of shape [batch, 3, 3].

    Returns:
        D: A `Tensor` of shape [batch].
    """
    pos_terms = [
        (0, 1, 2),
        (2, 0, 1),
        (1, 2, 0)
    ]
    neg_terms = [
        (2, 1, 0),
        (1, 0, 2),
        (0, 2, 1)
    ]
    pos_terms = _tf.add_n([M[:, 0, i] * M[:, 1, j] * M[:, 2, k]
        for i, j, k in pos_terms])
    neg_terms = _tf.add_n([M[:, 0, i] * M[:, 1, j] * M[:, 2, k]
        for i, j, k in neg_terms])
    return pos_terms - neg_terms
