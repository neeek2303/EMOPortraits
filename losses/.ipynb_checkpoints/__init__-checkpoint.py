from .adversarial import AdversarialLoss
from .feature_matching import FeatureMatchingLoss
from .head_pose_matching import HeadPoseMatchingLoss
from .perceptual import PerceptualLoss
from .segmentation import SegmentationLoss
from .equivariance import EquivarianceLoss

from .psnr import PSNR
from .lpips import LPIPS
from .gaze import GazeLoss
from .affine_params_matching import AffineLoss


from pytorch_msssim import SSIM, MS_SSIM