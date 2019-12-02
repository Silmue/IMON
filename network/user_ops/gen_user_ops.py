"""Python wrappers around TensorFlow ops.

This file is MACHINE GENERATED! Do not edit.
"""

import collections as _collections

from tensorflow.python.eager import execute as _execute
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import tensor_shape as _tensor_shape

from tensorflow.core.framework import op_def_pb2 as _op_def_pb2
# Needed to trigger the call to _set_call_cpp_shape_fn.
from tensorflow.python.framework import common_shapes as _common_shapes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library


def affine_flow(w, b, len1, len2, len3, name=None):
  r"""Returns an affine flow.

  Args:
    w: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    b: A `Tensor`. Must have the same type as `w`.
    len1: An `int`.
    len2: An `int`.
    len3: An `int`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `w`.
  """
  len1 = _execute.make_int(len1, "len1")
  len2 = _execute.make_int(len2, "len2")
  len3 = _execute.make_int(len3, "len3")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "AffineFlow", w=w, b=b, len1=len1, len2=len2, len3=len3, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "len1", _op.get_attr("len1"), "len2",
              _op.get_attr("len2"), "len3", _op.get_attr("len3"))
  else:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([w, b], _ctx, _dtypes.float32)
    (w, b) = _inputs_T
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [w, b]
    _attrs = ("T", _attr_T, "len1", len1, "len2", len2, "len3", len3)
    _result = _execute.execute(b"AffineFlow", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "AffineFlow", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


_affine_flow_gradient_outputs = ["dw", "db"]
_AffineFlowGradientOutput = _collections.namedtuple(
    "AffineFlowGradient", _affine_flow_gradient_outputs)


def affine_flow_gradient(dflow, len1, len2, len3, name=None):
  r"""The gradient of `AffineFlow` op.

  Args:
    dflow: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    len1: An `int`.
    len2: An `int`.
    len3: An `int`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (dw, db).

    dw: A `Tensor`. Has the same type as `dflow`.
    db: A `Tensor`. Has the same type as `dflow`.
  """
  len1 = _execute.make_int(len1, "len1")
  len2 = _execute.make_int(len2, "len2")
  len3 = _execute.make_int(len3, "len3")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "AffineFlowGradient", dflow=dflow, len1=len1, len2=len2, len3=len3,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "len1", _op.get_attr("len1"), "len2",
              _op.get_attr("len2"), "len3", _op.get_attr("len3"))
  else:
    _attr_T, (dflow,) = _execute.args_to_matching_eager([dflow], _ctx, _dtypes.float32)
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [dflow]
    _attrs = ("T", _attr_T, "len1", len1, "len2", len2, "len3", len3)
    _result = _execute.execute(b"AffineFlowGradient", 2, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "AffineFlowGradient", _inputs_flat, _attrs, _result, name)
  _result = _AffineFlowGradientOutput._make(_result)
  return _result


def correlation(img1, img2, kernel, displacement, stride, normalise=True, name=None):
  r"""Computes the correlation between two 3D images.

  Args:
    img1: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    img2: A `Tensor`. Must have the same type as `img1`.
    kernel: An `int`.
    displacement: An `int`.
    stride: An `int`.
    normalise: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `img1`.
  """
  kernel = _execute.make_int(kernel, "kernel")
  displacement = _execute.make_int(displacement, "displacement")
  stride = _execute.make_int(stride, "stride")
  if normalise is None:
    normalise = True
  normalise = _execute.make_bool(normalise, "normalise")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Correlation", img1=img1, img2=img2, kernel=kernel,
        displacement=displacement, stride=stride, normalise=normalise,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "kernel", _op.get_attr("kernel"),
              "displacement", _op.get_attr("displacement"), "stride",
              _op.get_attr("stride"), "normalise", _op.get_attr("normalise"))
  else:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([img1, img2], _ctx, _dtypes.float32)
    (img1, img2) = _inputs_T
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [img1, img2]
    _attrs = ("T", _attr_T, "kernel", kernel, "displacement", displacement,
              "stride", stride, "normalise", normalise)
    _result = _execute.execute(b"Correlation", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "Correlation", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


_correlation_gradient_outputs = ["dimg1", "dimg2"]
_CorrelationGradientOutput = _collections.namedtuple(
    "CorrelationGradient", _correlation_gradient_outputs)


def correlation_gradient(img1, img2, dcorr, kernel, displacement, stride, normalise, name=None):
  r"""The gradient of `Correlation` op.

  Args:
    img1: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    img2: A `Tensor`. Must have the same type as `img1`.
    dcorr: A `Tensor`. Must have the same type as `img1`.
    kernel: An `int`.
    displacement: An `int`.
    stride: An `int`.
    normalise: A `bool`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (dimg1, dimg2).

    dimg1: A `Tensor`. Has the same type as `img1`.
    dimg2: A `Tensor`. Has the same type as `img1`.
  """
  kernel = _execute.make_int(kernel, "kernel")
  displacement = _execute.make_int(displacement, "displacement")
  stride = _execute.make_int(stride, "stride")
  normalise = _execute.make_bool(normalise, "normalise")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "CorrelationGradient", img1=img1, img2=img2, dcorr=dcorr,
        kernel=kernel, displacement=displacement, stride=stride,
        normalise=normalise, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "kernel", _op.get_attr("kernel"),
              "displacement", _op.get_attr("displacement"), "stride",
              _op.get_attr("stride"), "normalise", _op.get_attr("normalise"))
  else:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([img1, img2, dcorr], _ctx, _dtypes.float32)
    (img1, img2, dcorr) = _inputs_T
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [img1, img2, dcorr]
    _attrs = ("T", _attr_T, "kernel", kernel, "displacement", displacement,
              "stride", stride, "normalise", normalise)
    _result = _execute.execute(b"CorrelationGradient", 2, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "CorrelationGradient", _inputs_flat, _attrs, _result, name)
  _result = _CorrelationGradientOutput._make(_result)
  return _result


_covariance_internal_outputs = ["x_sum", "y_sum", "x_mean", "y_mean",
                               "x_var_unscaled", "y_var_unscaled",
                               "xy_covar_unscaled", "x_var", "y_var",
                               "xy_covar", "xy_pearson"]
_CovarianceInternalOutput = _collections.namedtuple(
    "CovarianceInternal", _covariance_internal_outputs)


def covariance_internal(x, y, epsilon=1e-12, name=None):
  r"""Computes covariance and other statistics in batch.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    y: A `Tensor`. Must have the same type as `x`.
    epsilon: An optional `float`. Defaults to `1e-12`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (x_sum, y_sum, x_mean, y_mean, x_var_unscaled, y_var_unscaled, xy_covar_unscaled, x_var, y_var, xy_covar, xy_pearson).

    x_sum: A `Tensor`. Has the same type as `x`.
    y_sum: A `Tensor`. Has the same type as `x`.
    x_mean: A `Tensor`. Has the same type as `x`.
    y_mean: A `Tensor`. Has the same type as `x`.
    x_var_unscaled: A `Tensor`. Has the same type as `x`.
    y_var_unscaled: A `Tensor`. Has the same type as `x`.
    xy_covar_unscaled: A `Tensor`. Has the same type as `x`.
    x_var: A `Tensor`. Has the same type as `x`.
    y_var: A `Tensor`. Has the same type as `x`.
    xy_covar: A `Tensor`. Has the same type as `x`.
    xy_pearson: A `Tensor`. Has the same type as `x`.
  """
  if epsilon is None:
    epsilon = 1e-12
  epsilon = _execute.make_float(epsilon, "epsilon")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "CovarianceInternal", x=x, y=y, epsilon=epsilon, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "epsilon", _op.get_attr("epsilon"))
  else:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y], _ctx, _dtypes.float32)
    (x, y) = _inputs_T
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [x, y]
    _attrs = ("T", _attr_T, "epsilon", epsilon)
    _result = _execute.execute(b"CovarianceInternal", 11, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "CovarianceInternal", _inputs_flat, _attrs, _result, name)
  _result = _CovarianceInternalOutput._make(_result)
  return _result


_covariance_internal_gradient_outputs = ["dx", "dy"]
_CovarianceInternalGradientOutput = _collections.namedtuple(
    "CovarianceInternalGradient", _covariance_internal_gradient_outputs)


def covariance_internal_gradient(x, y, x_mean, y_mean, x_var_unscaled, y_var_unscaled, xy_pearson, dx_sum, dy_sum, dx_mean, dy_mean, dx_var_unscaled, dy_var_unscaled, dxy_covar_unscaled, dx_var, dy_var, dxy_covar, dxy_pearson, epsilon, name=None):
  r"""The gradient of `CovarianceInternal` op.

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    y: A `Tensor`. Must have the same type as `x`.
    x_mean: A `Tensor`. Must have the same type as `x`.
    y_mean: A `Tensor`. Must have the same type as `x`.
    x_var_unscaled: A `Tensor`. Must have the same type as `x`.
    y_var_unscaled: A `Tensor`. Must have the same type as `x`.
    xy_pearson: A `Tensor`. Must have the same type as `x`.
    dx_sum: A `Tensor`. Must have the same type as `x`.
    dy_sum: A `Tensor`. Must have the same type as `x`.
    dx_mean: A `Tensor`. Must have the same type as `x`.
    dy_mean: A `Tensor`. Must have the same type as `x`.
    dx_var_unscaled: A `Tensor`. Must have the same type as `x`.
    dy_var_unscaled: A `Tensor`. Must have the same type as `x`.
    dxy_covar_unscaled: A `Tensor`. Must have the same type as `x`.
    dx_var: A `Tensor`. Must have the same type as `x`.
    dy_var: A `Tensor`. Must have the same type as `x`.
    dxy_covar: A `Tensor`. Must have the same type as `x`.
    dxy_pearson: A `Tensor`. Must have the same type as `x`.
    epsilon: A `float`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (dx, dy).

    dx: A `Tensor`. Has the same type as `x`.
    dy: A `Tensor`. Has the same type as `x`.
  """
  epsilon = _execute.make_float(epsilon, "epsilon")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "CovarianceInternalGradient", x=x, y=y, x_mean=x_mean, y_mean=y_mean,
        x_var_unscaled=x_var_unscaled, y_var_unscaled=y_var_unscaled,
        xy_pearson=xy_pearson, dx_sum=dx_sum, dy_sum=dy_sum, dx_mean=dx_mean,
        dy_mean=dy_mean, dx_var_unscaled=dx_var_unscaled,
        dy_var_unscaled=dy_var_unscaled,
        dxy_covar_unscaled=dxy_covar_unscaled, dx_var=dx_var, dy_var=dy_var,
        dxy_covar=dxy_covar, dxy_pearson=dxy_pearson, epsilon=epsilon,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "epsilon", _op.get_attr("epsilon"))
  else:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([x, y, x_mean, y_mean, x_var_unscaled, y_var_unscaled, xy_pearson, dx_sum, dy_sum, dx_mean, dy_mean, dx_var_unscaled, dy_var_unscaled, dxy_covar_unscaled, dx_var, dy_var, dxy_covar, dxy_pearson], _ctx, _dtypes.float32)
    (x, y, x_mean, y_mean, x_var_unscaled, y_var_unscaled, xy_pearson, dx_sum, dy_sum, dx_mean, dy_mean, dx_var_unscaled, dy_var_unscaled, dxy_covar_unscaled, dx_var, dy_var, dxy_covar, dxy_pearson) = _inputs_T
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [x, y, x_mean, y_mean, x_var_unscaled, y_var_unscaled, xy_pearson, dx_sum, dy_sum, dx_mean, dy_mean, dx_var_unscaled, dy_var_unscaled, dxy_covar_unscaled, dx_var, dy_var, dxy_covar, dxy_pearson]
    _attrs = ("T", _attr_T, "epsilon", epsilon)
    _result = _execute.execute(b"CovarianceInternalGradient", 2,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "CovarianceInternalGradient", _inputs_flat, _attrs, _result, name)
  _result = _CovarianceInternalGradientOutput._make(_result)
  return _result


def fast_leaky_relu(in_, leaking=0.1, name=None):
  r"""Computes leaky ReLU efficiently.

  Args:
    in_: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    leaking: An optional `float`. Defaults to `0.1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `in_`.
  """
  if leaking is None:
    leaking = 0.1
  leaking = _execute.make_float(leaking, "leaking")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "FastLeakyRelu", in_=in_, leaking=leaking, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "leaking", _op.get_attr("leaking"))
  else:
    _attr_T, (in_,) = _execute.args_to_matching_eager([in_], _ctx, _dtypes.float32)
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [in_]
    _attrs = ("T", _attr_T, "leaking", leaking)
    _result = _execute.execute(b"FastLeakyRelu", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "FastLeakyRelu", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def fast_leaky_relu_gradient(in_, dout, leaking, name=None):
  r"""The gradient of `FastLeakyRelu` op.

  Args:
    in_: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    dout: A `Tensor`. Must have the same type as `in_`.
    leaking: A `float`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `in_`.
  """
  leaking = _execute.make_float(leaking, "leaking")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "FastLeakyReluGradient", in_=in_, dout=dout, leaking=leaking,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "leaking", _op.get_attr("leaking"))
  else:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([in_, dout], _ctx, _dtypes.float32)
    (in_, dout) = _inputs_T
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [in_, dout]
    _attrs = ("T", _attr_T, "leaking", leaking)
    _result = _execute.execute(b"FastLeakyReluGradient", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "FastLeakyReluGradient", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def l1_matrix(in1, in2, amplifier, name=None):
  r"""Computes pairwise L1 distance.

  Args:
    in1: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    in2: A `Tensor`. Must have the same type as `in1`.
    amplifier: A `float`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `in1`.
  """
  amplifier = _execute.make_float(amplifier, "amplifier")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "L1Matrix", in1=in1, in2=in2, amplifier=amplifier, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "amplifier", _op.get_attr("amplifier"))
  else:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([in1, in2], _ctx, _dtypes.float32)
    (in1, in2) = _inputs_T
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [in1, in2]
    _attrs = ("T", _attr_T, "amplifier", amplifier)
    _result = _execute.execute(b"L1Matrix", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "L1Matrix", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


_l1_matrix_gradient_outputs = ["din1", "din2"]
_L1MatrixGradientOutput = _collections.namedtuple(
    "L1MatrixGradient", _l1_matrix_gradient_outputs)


def l1_matrix_gradient(in1, in2, dout, amplifier, name=None):
  r"""The gradient of `L1Matrix` op.

  Args:
    in1: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    in2: A `Tensor`. Must have the same type as `in1`.
    dout: A `Tensor`. Must have the same type as `in1`.
    amplifier: A `float`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (din1, din2).

    din1: A `Tensor`. Has the same type as `in1`.
    din2: A `Tensor`. Has the same type as `in1`.
  """
  amplifier = _execute.make_float(amplifier, "amplifier")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "L1MatrixGradient", in1=in1, in2=in2, dout=dout, amplifier=amplifier,
        name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "amplifier", _op.get_attr("amplifier"))
  else:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([in1, in2, dout], _ctx, _dtypes.float32)
    (in1, in2, dout) = _inputs_T
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [in1, in2, dout]
    _attrs = ("T", _attr_T, "amplifier", amplifier)
    _result = _execute.execute(b"L1MatrixGradient", 2, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "L1MatrixGradient", _inputs_flat, _attrs, _result, name)
  _result = _L1MatrixGradientOutput._make(_result)
  return _result


def reconstruction(img2, flow, coef1=1, coef2=1, coef3=1, name=None):
  r"""Reconstructs the first image from the second image and the flow (1 -> 2).

  Args:
    img2: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    flow: A `Tensor`. Must have the same type as `img2`.
    coef1: An optional `float`. Defaults to `1`.
    coef2: An optional `float`. Defaults to `1`.
    coef3: An optional `float`. Defaults to `1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `img2`.
  """
  if coef1 is None:
    coef1 = 1
  coef1 = _execute.make_float(coef1, "coef1")
  if coef2 is None:
    coef2 = 1
  coef2 = _execute.make_float(coef2, "coef2")
  if coef3 is None:
    coef3 = 1
  coef3 = _execute.make_float(coef3, "coef3")
  _ctx = _context.context()
  #if _ctx.in_graph_mode():
  if not _ctx.executing_eagerly():
    _, _, _op = _op_def_lib._apply_op_helper(
        "Reconstruction", img2=img2, flow=flow, coef1=coef1, coef2=coef2,
        coef3=coef3, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "coef1", _op.get_attr("coef1"), "coef2",
              _op.get_attr("coef2"), "coef3", _op.get_attr("coef3"))
  else:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([img2, flow], _ctx, _dtypes.float32)
    (img2, flow) = _inputs_T
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [img2, flow]
    _attrs = ("T", _attr_T, "coef1", coef1, "coef2", coef2, "coef3", coef3)
    _result = _execute.execute(b"Reconstruction", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "Reconstruction", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def reconstruction_gradient(img2, flow, dimg1, coef1, coef2, coef3, name=None):
  r"""The gradient of `Reconstruction` op with respect to flow.

  Args:
    img2: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    flow: A `Tensor`. Must have the same type as `img2`.
    dimg1: A `Tensor`. Must have the same type as `img2`.
    coef1: A `float`.
    coef2: A `float`.
    coef3: A `float`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `img2`.
  """
  coef1 = _execute.make_float(coef1, "coef1")
  coef2 = _execute.make_float(coef2, "coef2")
  coef3 = _execute.make_float(coef3, "coef3")
  _ctx = _context.context()
  #if _ctx.in_graph_mode():
  if not _ctx.executing_eagerly():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ReconstructionGradient", img2=img2, flow=flow, dimg1=dimg1,
        coef1=coef1, coef2=coef2, coef3=coef3, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "coef1", _op.get_attr("coef1"), "coef2",
              _op.get_attr("coef2"), "coef3", _op.get_attr("coef3"))
  else:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([img2, flow, dimg1], _ctx, _dtypes.float32)
    (img2, flow, dimg1) = _inputs_T
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [img2, flow, dimg1]
    _attrs = ("T", _attr_T, "coef1", coef1, "coef2", coef2, "coef3", coef3)
    _result = _execute.execute(b"ReconstructionGradient", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "ReconstructionGradient", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def reconstruction_wrt_image_gradient(img2, flow, dimg1, coef1, coef2, coef3, name=None):
  r"""The gradient of `Reconstruction` op with respect to img2.

  Args:
    img2: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    flow: A `Tensor`. Must have the same type as `img2`.
    dimg1: A `Tensor`. Must have the same type as `img2`.
    coef1: A `float`.
    coef2: A `float`.
    coef3: A `float`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `img2`.
  """
  coef1 = _execute.make_float(coef1, "coef1")
  coef2 = _execute.make_float(coef2, "coef2")
  coef3 = _execute.make_float(coef3, "coef3")
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ReconstructionWrtImageGradient", img2=img2, flow=flow, dimg1=dimg1,
        coef1=coef1, coef2=coef2, coef3=coef3, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"), "coef1", _op.get_attr("coef1"), "coef2",
              _op.get_attr("coef2"), "coef3", _op.get_attr("coef3"))
  else:
    _attr_T, _inputs_T = _execute.args_to_matching_eager([img2, flow, dimg1], _ctx, _dtypes.float32)
    (img2, flow, dimg1) = _inputs_T
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [img2, flow, dimg1]
    _attrs = ("T", _attr_T, "coef1", coef1, "coef2", coef2, "coef3", coef3)
    _result = _execute.execute(b"ReconstructionWrtImageGradient", 1,
                               inputs=_inputs_flat, attrs=_attrs, ctx=_ctx,
                               name=name)
  _execute.record_gradient(
      "ReconstructionWrtImageGradient", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result


def reduce_sum_two_dim(in_, name=None):
  r"""Reduces the second dimension of a rank-2 `Tensor`.

  Args:
    in_: A `Tensor`. Must be one of the following types: `float32`, `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `in_`.
  """
  _ctx = _context.context()
  if _ctx.in_graph_mode():
    _, _, _op = _op_def_lib._apply_op_helper(
        "ReduceSumTwoDim", in_=in_, name=name)
    _result = _op.outputs[:]
    _inputs_flat = _op.inputs
    _attrs = ("T", _op.get_attr("T"))
  else:
    _attr_T, (in_,) = _execute.args_to_matching_eager([in_], _ctx, _dtypes.float32)
    _attr_T = _attr_T.as_datatype_enum
    _inputs_flat = [in_]
    _attrs = ("T", _attr_T)
    _result = _execute.execute(b"ReduceSumTwoDim", 1, inputs=_inputs_flat,
                               attrs=_attrs, ctx=_ctx, name=name)
  _execute.record_gradient(
      "ReduceSumTwoDim", _inputs_flat, _attrs, _result, name)
  _result, = _result
  return _result

def _InitOpDefLibrary(op_list_proto_bytes):
  op_list = _op_def_pb2.OpList()
  op_list.ParseFromString(op_list_proto_bytes)
  _op_def_registry.register_op_list(op_list)
  op_def_lib = _op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib
# op {
#   name: "AffineFlow"
#   input_arg {
#     name: "w"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "b"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "flow"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "len1"
#     type: "int"
#   }
#   attr {
#     name: "len2"
#     type: "int"
#   }
#   attr {
#     name: "len3"
#     type: "int"
#   }
# }
# op {
#   name: "AffineFlowGradient"
#   input_arg {
#     name: "dflow"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "dw"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "db"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "len1"
#     type: "int"
#   }
#   attr {
#     name: "len2"
#     type: "int"
#   }
#   attr {
#     name: "len3"
#     type: "int"
#   }
# }
# op {
#   name: "Correlation"
#   input_arg {
#     name: "img1"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "img2"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "corr"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "kernel"
#     type: "int"
#   }
#   attr {
#     name: "displacement"
#     type: "int"
#   }
#   attr {
#     name: "stride"
#     type: "int"
#   }
#   attr {
#     name: "normalise"
#     type: "bool"
#     default_value {
#       b: true
#     }
#   }
# }
# op {
#   name: "CorrelationGradient"
#   input_arg {
#     name: "img1"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "img2"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "dcorr"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "dimg1"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "dimg2"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "kernel"
#     type: "int"
#   }
#   attr {
#     name: "displacement"
#     type: "int"
#   }
#   attr {
#     name: "stride"
#     type: "int"
#   }
#   attr {
#     name: "normalise"
#     type: "bool"
#   }
# }
# op {
#   name: "CovarianceInternal"
#   input_arg {
#     name: "x"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "y"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "x_sum"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "y_sum"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "x_mean"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "y_mean"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "x_var_unscaled"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "y_var_unscaled"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "xy_covar_unscaled"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "x_var"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "y_var"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "xy_covar"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "xy_pearson"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "epsilon"
#     type: "float"
#     default_value {
#       f: 1e-12
#     }
#   }
# }
# op {
#   name: "CovarianceInternalGradient"
#   input_arg {
#     name: "x"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "y"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "x_mean"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "y_mean"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "x_var_unscaled"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "y_var_unscaled"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "xy_pearson"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "dx_sum"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "dy_sum"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "dx_mean"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "dy_mean"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "dx_var_unscaled"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "dy_var_unscaled"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "dxy_covar_unscaled"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "dx_var"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "dy_var"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "dxy_covar"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "dxy_pearson"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "dx"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "dy"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "epsilon"
#     type: "float"
#   }
# }
# op {
#   name: "FastLeakyRelu"
#   input_arg {
#     name: "in"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "out"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "leaking"
#     type: "float"
#     default_value {
#       f: 0.1
#     }
#   }
# }
# op {
#   name: "FastLeakyReluGradient"
#   input_arg {
#     name: "in"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "dout"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "din"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "leaking"
#     type: "float"
#   }
# }
# op {
#   name: "L1Matrix"
#   input_arg {
#     name: "in1"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "in2"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "out"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "amplifier"
#     type: "float"
#   }
# }
# op {
#   name: "L1MatrixGradient"
#   input_arg {
#     name: "in1"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "in2"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "dout"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "din1"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "din2"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "amplifier"
#     type: "float"
#   }
# }
# op {
#   name: "Reconstruction"
#   input_arg {
#     name: "img2"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "flow"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "img1"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "coef1"
#     type: "float"
#     default_value {
#       f: 1
#     }
#   }
#   attr {
#     name: "coef2"
#     type: "float"
#     default_value {
#       f: 1
#     }
#   }
#   attr {
#     name: "coef3"
#     type: "float"
#     default_value {
#       f: 1
#     }
#   }
# }
# op {
#   name: "ReconstructionGradient"
#   input_arg {
#     name: "img2"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "flow"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "dimg1"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "dflow"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "coef1"
#     type: "float"
#   }
#   attr {
#     name: "coef2"
#     type: "float"
#   }
#   attr {
#     name: "coef3"
#     type: "float"
#   }
# }
# op {
#   name: "ReconstructionWrtImageGradient"
#   input_arg {
#     name: "img2"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "flow"
#     type_attr: "T"
#   }
#   input_arg {
#     name: "dimg1"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "dimg2"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
#   attr {
#     name: "coef1"
#     type: "float"
#   }
#   attr {
#     name: "coef2"
#     type: "float"
#   }
#   attr {
#     name: "coef3"
#     type: "float"
#   }
# }
# op {
#   name: "ReduceSumTwoDim"
#   input_arg {
#     name: "in"
#     type_attr: "T"
#   }
#   output_arg {
#     name: "out"
#     type_attr: "T"
#   }
#   attr {
#     name: "T"
#     type: "type"
#     default_value {
#       type: DT_FLOAT
#     }
#     allowed_values {
#       list {
#         type: DT_FLOAT
#         type: DT_DOUBLE
#       }
#     }
#   }
# }
_op_def_lib = _InitOpDefLibrary(b"\ne\n\nAffineFlow\022\006\n\001w\"\001T\022\006\n\001b\"\001T\032\t\n\004flow\"\001T\"\025\n\001T\022\004type\032\0020\001:\006\n\0042\002\001\002\"\013\n\004len1\022\003int\"\013\n\004len2\022\003int\"\013\n\004len3\022\003int\np\n\022AffineFlowGradient\022\n\n\005dflow\"\001T\032\007\n\002dw\"\001T\032\007\n\002db\"\001T\"\025\n\001T\022\004type\032\0020\001:\006\n\0042\002\001\002\"\013\n\004len1\022\003int\"\013\n\004len2\022\003int\"\013\n\004len3\022\003int\n\217\001\n\013Correlation\022\t\n\004img1\"\001T\022\t\n\004img2\"\001T\032\t\n\004corr\"\001T\"\025\n\001T\022\004type\032\0020\001:\006\n\0042\002\001\002\"\r\n\006kernel\022\003int\"\023\n\014displacement\022\003int\"\r\n\006stride\022\003int\"\025\n\tnormalise\022\004bool\032\002(\001\n\254\001\n\023CorrelationGradient\022\t\n\004img1\"\001T\022\t\n\004img2\"\001T\022\n\n\005dcorr\"\001T\032\n\n\005dimg1\"\001T\032\n\n\005dimg2\"\001T\"\025\n\001T\022\004type\032\0020\001:\006\n\0042\002\001\002\"\r\n\006kernel\022\003int\"\023\n\014displacement\022\003int\"\r\n\006stride\022\003int\"\021\n\tnormalise\022\004bool\n\200\002\n\022CovarianceInternal\022\006\n\001x\"\001T\022\006\n\001y\"\001T\032\n\n\005x_sum\"\001T\032\n\n\005y_sum\"\001T\032\013\n\006x_mean\"\001T\032\013\n\006y_mean\"\001T\032\023\n\016x_var_unscaled\"\001T\032\023\n\016y_var_unscaled\"\001T\032\026\n\021xy_covar_unscaled\"\001T\032\n\n\005x_var\"\001T\032\n\n\005y_var\"\001T\032\r\n\010xy_covar\"\001T\032\017\n\nxy_pearson\"\001T\"\025\n\001T\022\004type\032\0020\001:\006\n\0042\002\001\002\"\027\n\007epsilon\022\005float\032\005%\314\274\214+\n\363\002\n\032CovarianceInternalGradient\022\006\n\001x\"\001T\022\006\n\001y\"\001T\022\013\n\006x_mean\"\001T\022\013\n\006y_mean\"\001T\022\023\n\016x_var_unscaled\"\001T\022\023\n\016y_var_unscaled\"\001T\022\017\n\nxy_pearson\"\001T\022\013\n\006dx_sum\"\001T\022\013\n\006dy_sum\"\001T\022\014\n\007dx_mean\"\001T\022\014\n\007dy_mean\"\001T\022\024\n\017dx_var_unscaled\"\001T\022\024\n\017dy_var_unscaled\"\001T\022\027\n\022dxy_covar_unscaled\"\001T\022\013\n\006dx_var\"\001T\022\013\n\006dy_var\"\001T\022\016\n\tdxy_covar\"\001T\022\020\n\013dxy_pearson\"\001T\032\007\n\002dx\"\001T\032\007\n\002dy\"\001T\"\025\n\001T\022\004type\032\0020\001:\006\n\0042\002\001\002\"\020\n\007epsilon\022\005float\nR\n\rFastLeakyRelu\022\007\n\002in\"\001T\032\010\n\003out\"\001T\"\025\n\001T\022\004type\032\0020\001:\006\n\0042\002\001\002\"\027\n\007leaking\022\005float\032\005%\315\314\314=\n^\n\025FastLeakyReluGradient\022\007\n\002in\"\001T\022\t\n\004dout\"\001T\032\010\n\003din\"\001T\"\025\n\001T\022\004type\032\0020\001:\006\n\0042\002\001\002\"\020\n\007leaking\022\005float\nS\n\010L1Matrix\022\010\n\003in1\"\001T\022\010\n\003in2\"\001T\032\010\n\003out\"\001T\"\025\n\001T\022\004type\032\0020\001:\006\n\0042\002\001\002\"\022\n\tamplifier\022\005float\nr\n\020L1MatrixGradient\022\010\n\003in1\"\001T\022\010\n\003in2\"\001T\022\t\n\004dout\"\001T\032\t\n\004din1\"\001T\032\t\n\004din2\"\001T\"\025\n\001T\022\004type\032\0020\001:\006\n\0042\002\001\002\"\022\n\tamplifier\022\005float\n\215\001\n\016Reconstruction\022\t\n\004img2\"\001T\022\t\n\004flow\"\001T\032\t\n\004img1\"\001T\"\025\n\001T\022\004type\032\0020\001:\006\n\0042\002\001\002\"\025\n\005coef1\022\005float\032\005%\000\000\200?\"\025\n\005coef2\022\005float\032\005%\000\000\200?\"\025\n\005coef3\022\005float\032\005%\000\000\200?\n\215\001\n\026ReconstructionGradient\022\t\n\004img2\"\001T\022\t\n\004flow\"\001T\022\n\n\005dimg1\"\001T\032\n\n\005dflow\"\001T\"\025\n\001T\022\004type\032\0020\001:\006\n\0042\002\001\002\"\016\n\005coef1\022\005float\"\016\n\005coef2\022\005float\"\016\n\005coef3\022\005float\n\225\001\n\036ReconstructionWrtImageGradient\022\t\n\004img2\"\001T\022\t\n\004flow\"\001T\022\n\n\005dimg1\"\001T\032\n\n\005dimg2\"\001T\"\025\n\001T\022\004type\032\0020\001:\006\n\0042\002\001\002\"\016\n\005coef1\022\005float\"\016\n\005coef2\022\005float\"\016\n\005coef3\022\005float\n;\n\017ReduceSumTwoDim\022\007\n\002in\"\001T\032\010\n\003out\"\001T\"\025\n\001T\022\004type\032\0020\001:\006\n\0042\002\001\002")
