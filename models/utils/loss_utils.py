import torch
import torch.nn as nn 
import torch.nn.functional as F


def xentropy_loss(true, pred, weights=None, reduction=True):
    """Cross entropy loss.

    Args:
        pred: prediction array
        true: ground truth array

    Returns:
        cross entropy loss

    """
    true = torch.squeeze(true).type(torch.int64)
    # assert len(true.shape) == 3
    reduction = 'none' if not reduction else 'mean'
    loss = F.cross_entropy(pred, true, weights, reduction=reduction)
    return loss


def focal_loss(true, pred, alpha=None, gamma=2, reduction=True):
    """Focal loss.

    Args:
        pred: prediction array
        true: ground truth array
        alpha: weighting factor
        gamma: focal constant

    Returns:
        cross entropy loss

    """
    true = torch.squeeze(true).type(torch.int64)
    # assert len(true.shape) == 3
    
    log_p = F.log_softmax(pred, dim=-1)
    ce = F.nll_loss(log_p, true, alpha)

    # get true class column from each row
    all_rows = torch.arange(len(pred))
    log_pt = log_p[all_rows, true]

    # compute focal term: (1 - pt)^gamma
    pt = log_pt.exp()
    focal_term = (1 - pt)**gamma

    # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
    loss = focal_term * ce

    if reduction:
        loss = loss.mean()

    return loss

# TODO: add weights and reduction mode
def dice_loss(true, pred, reduction=None, smooth=1e-3, mask=None):
    """`pred` and `true` must be of torch.float32
    Assuming of shape NxHxWxC.

    """
    if mask is None:
        inse = torch.sum(pred * true, (0, 2, 3))
        l = torch.sum(pred, (0, 2, 3))
        r = torch.sum(true, (0, 2, 3))
    else:
        inse = torch.sum(pred * true * mask, (0, 2, 3))
        l = torch.sum(pred * mask, (0, 2, 3))
        r = torch.sum(true * mask, (0, 2, 3))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = torch.sum(loss)
    return loss


def mse_loss(true, pred, reduction=True):
    """Calculate mean squared error loss.

    Args:
        true: ground truth of combined horizontal
              and vertical maps
        pred: prediction of combined horizontal
              and vertical maps

    Returns:
        loss: mean squared error

    """
    loss = pred - true
    loss = (loss * loss)
    if reduction:
        return loss.mean()
    return loss

# TODO: add reduction mode
def msge_loss(true, pred, focus):
    """Calculate the mean squared error of the gradients of
    horizontal and vertical map predictions. Assumes
    channel 0 is Vertical and channel 1 is Horizontal.

    Args:
        true:  ground truth of combined horizontal
               and vertical maps
        pred:  prediction of combined horizontal
               and vertical maps
        focus: area where to apply loss (we only calculate
                the loss within the nuclei)

    Returns:
        loss:  mean squared error of gradients

    """

    def get_sobel_kernel(size):
        """Get sobel kernel with a given size."""
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device="cuda",
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range)
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    def get_gradient_hv(hv):
        """For calculating gradient."""
        kernel_h, kernel_v = get_sobel_kernel(5)
        kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
        kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

        h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
        v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

        # can only apply in NCHW mode
        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
        return dhv

    focus = (focus[..., None]).float()  # assume input NHW
    focus = torch.cat([focus, focus], axis=-1)
    true_grad = get_gradient_hv(true)
    pred_grad = get_gradient_hv(pred)
    loss = pred_grad - true_grad
    loss = focus * (loss * loss)
    # artificial reduce_mean with focused region
    loss = loss.sum() / (focus.sum() + 1.0e-8)
    return loss


def simclr_loss(features, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR.

    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    mask = torch.eye(batch_size, dtype=torch.float32).to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temperature / base_temperature) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss