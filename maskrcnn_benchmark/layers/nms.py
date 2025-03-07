# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from maskrcnn_benchmark import _C

from apex import amp

#amp.register_float_function(_C, 'nms')

#nms = _C.nms
# Only valid with fp32 inputs - give AMP the hint
nms = amp.float_function(_C.nms)
#nms = amp.promote_function(_C.nms)
# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
