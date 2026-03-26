import lpips
import torch
import torch.nn as nn

from model.vgg_arch import VGGFeatureExtractor


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(
        self,
        layer_weights,
        vgg_type="vgg19",
        use_input_norm=True,
        range_norm=False,
        perceptual_weight=1.0,
        style_weight=0.0,
        criterion="l1",
    ):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm,
        )

        self.criterion_type = criterion
        if self.criterion_type == "l1":
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == "mse":
            self.criterion = torch.nn.MSELoss(reduction="mean")
        elif self.criterion_type == "fro":
            self.criterion = None
        else:
            raise NotImplementedError(f"{criterion} criterion has not been supported.")

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == "fro":
                    percep_loss += (
                        torch.norm(x_features[k] - gt_features[k], p="fro")
                        * self.layer_weights[k]
                    )
                else:
                    percep_loss += (
                        self.criterion(x_features[k], gt_features[k])
                        * self.layer_weights[k]
                    )
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == "fro":
                    style_loss += (
                        torch.norm(
                            self._gram_mat(x_features[k])
                            - self._gram_mat(gt_features[k]),
                            p="fro",
                        )
                        * self.layer_weights[k]
                    )
                else:
                    style_loss += (
                        self.criterion(
                            self._gram_mat(x_features[k]),
                            self._gram_mat(gt_features[k]),
                        )
                        * self.layer_weights[k]
                    )
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


class LPIPSLoss(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        use_input_norm=True,
        range_norm=False,
    ):
        super(LPIPSLoss, self).__init__()
        self.perceptual = lpips.LPIPS(net="vgg", spatial=False).eval()
        self.loss_weight = loss_weight
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                "mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            # the std is for image with range [0, 1]
            self.register_buffer(
                "std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )

    def forward(self, pred, target):
        if self.range_norm:
            pred = (pred + 1) / 2
            target = (target + 1) / 2
        if self.use_input_norm:
            pred = (pred - self.mean) / self.std
            target = (target - self.mean) / self.std
        lpips_loss = self.perceptual(target.contiguous(), pred.contiguous())
        return self.loss_weight * lpips_loss.mean(), None


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type="nsgan", target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))

        if type == "nsgan":
            self.criterion = nn.BCELoss()
        elif type == "lsgan":
            self.criterion = nn.MSELoss()
        elif type == "hinge":
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == "hinge":
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(
                outputs
            )
            loss = self.criterion(outputs, labels)
            return loss


# =============================================================================
# Focal Frequency Loss (Jiang et al., ICCV 2021)
# Applied ONLY to the high-frequency residual of the HOLE region.
# This avoids inter-frequency gradient conflict as described in DRCN (IJCAI 2025).
# =============================================================================
import torch.fft
import torch.nn.functional as F


def _gaussian_blur_for_ffl(x, kernel_size=5, sigma=1.0):
    """Depthwise Gaussian blur — extracts low-frequency component of x."""
    B, C, H, W = x.shape
    coords = torch.arange(kernel_size, device=x.device).float() - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]  # (k, k)
    kernel = kernel.expand(C, 1, kernel_size, kernel_size)  # (C, 1, k, k)
    pad = kernel_size // 2
    return F.conv2d(x, kernel, padding=pad, groups=C)


def laplacian_decompose_ffl(x, kernel_size=5, sigma=1.0):
    """
    Split frame into low-frequency and high-frequency components.
    Args:
        x (Tensor): (B, C, H, W) float tensor, any value range.
    Returns:
        low  (Tensor): blurred / low-frequency version of x.
        high (Tensor): residual = x - low  (edges, textures).
    """
    low = _gaussian_blur_for_ffl(x, kernel_size, sigma)
    high = x - low
    return low, high


class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss — operates on the HIGH-FREQUENCY component only.

    The focal weighting dynamically up-weights frequency components that
    the model reconstructs poorly, pushing training to focus on hard cases.

    Args:
        loss_weight (float): Scalar multiplier on final loss. Default: 1.0.
                             (Controlled externally via config ffl_weight.)
        alpha (float):       Focusing exponent. Higher = more aggressive
                             focus on poorly-reconstructed frequencies.
                             Default: 1.0 (from original paper).
        kernel_size (int):   Gaussian kernel size for decomposition. Default: 5.
        sigma (float):       Gaussian sigma for decomposition. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, kernel_size=5, sigma=1.0):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.sigma = sigma

    def _focal_freq_core(self, pred_freq, real_freq):
        """
        Focal-weighted frequency-domain L2 between pred and real.
        Args:
            pred_freq, real_freq: (..., H, W, 2) — stacked real/imag parts.
        """
        diff = (pred_freq - real_freq) ** 2
        freq_dist = diff[..., 0] + diff[..., 1]  # real² + imag²
        # Dynamic weight: frequencies with high error get higher weight
        weight = torch.exp(self.alpha * freq_dist.detach() ** 0.5)
        return torch.mean(weight * freq_dist)

    def forward(self, pred, target):
        """
        Args:
            pred   (Tensor): predicted hole region  (B, C, H, W), range [-1,1]
            target (Tensor): ground-truth hole region (B, C, H, W), range [-1,1]

        NOTE: Both pred and target should already be masked (multiplied by mask)
              before being passed in. Do NOT pass the full frame here.

        Returns:
            Scalar loss tensor.
        """
        # Step 1: Decompose → keep only HIGH-frequency residual
        _, pred_high = laplacian_decompose_ffl(pred, self.kernel_size, self.sigma)
        _, target_high = laplacian_decompose_ffl(target, self.kernel_size, self.sigma)

        # Step 2: 2-D FFT on spatial dimensions
        pred_f = torch.fft.fft2(pred_high, norm="ortho")
        target_f = torch.fft.fft2(target_high, norm="ortho")

        # Step 3: Shift DC component to center; stack real/imag → (..., H, W, 2)
        pred_f = torch.fft.fftshift(pred_f, dim=(-2, -1))
        target_f = torch.fft.fftshift(target_f, dim=(-2, -1))
        pred_f = torch.stack([pred_f.real, pred_f.imag], dim=-1)
        target_f = torch.stack([target_f.real, target_f.imag], dim=-1)

        # Step 4: Focal-weighted loss × overall weight
        return self._focal_freq_core(pred_f, target_f) * self.loss_weight
