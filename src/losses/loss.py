"""

loss function

"""

import torch
import numpy as np
from src.utils.utils_train import nanmean


def negative_log_likelihood(mean, log_var, labels):
    """
    Negative log likelihood of the data. Adding weight decay yields the negative log posterior. Note that we're
    interpreting all labels of a given pixel as individual training samples, corresponding to an assumed multivariate
    Gaussian with diagonal covariance matrix (i.e. independent label noise over tasks). Also, we only consider
    locations where labels != nan.
    """
    assert mean.shape == log_var.shape == labels.shape
    precision = torch.exp(-log_var)
    diff = mean - labels
    # we set any nan's to zero to prevent nan gradients in the backward pass. This also stops gradient flow through
    # the locations where labels are nan, but we don't need gradients for these locations anyway. Further info:
    # https://github.com/pytorch/pytorch/issues/15506, https://github.com/pytorch/pytorch/issues/15131
    diff[diff.isnan()] = 0.
    losses = precision * diff**2 + log_var
    return losses

def mseloss(preds, labels, weight=False, thresh=30.0):
    """
    compute standard mse for regression
    """
    diff = preds - labels
    diff[diff.isnan()] = 0.

    if weight:
        w = (labels - thresh)/2
        # not weight label <30 and >55
        w[w<1] = 1
        w[w>12] = 1
        losses = torch.mul((diff**2), w)
    else:
        losses = diff**2

    return losses

def maeloss(preds, labels):
    """
    compute standard mae for regression
    """
    diff = preds - labels
    diff[diff.isnan()] = 0.
    losses = torch.abs(diff)

    return losses
