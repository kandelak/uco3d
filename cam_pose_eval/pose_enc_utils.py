import numpy as np
import torch

from rotation_utils import mat_to_quat, quat_to_mat


def pose_encoding_to_extri_intri(
        pose_encoding, image_size_hw=None, pose_encoding_type="absT_quaR_FoV", build_intrinsics=True  # e.g., (256, 512)
):
    """Convert a pose encoding back to camera extrinsics and intrinsics.

    This function performs the inverse operation of extri_intri_to_pose_encoding,
    reconstructing the full camera parameters from the compact encoding.

    Args:
        pose_encoding (torch.Tensor): Encoded camera pose parameters with shape BxSx9,
            where B is batch size and S is sequence length.
            For "absT_quaR_FoV" type, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for reconstructing intrinsics from field of view values.
            For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding used. Currently only
            supports "absT_quaR_FoV" (absolute translation, quaternion rotation, field of view).
        build_intrinsics (bool): Whether to reconstruct the intrinsics matrix.
            If False, only extrinsics are returned and intrinsics will be None.

    Returns:
        tuple: (extrinsics, intrinsics)
            - extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4.
              In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world
              transformation. The format is [R|t] where R is a 3x3 rotation matrix and t is
              a 3x1 translation vector.
            - intrinsics (torch.Tensor or None): Camera intrinsic parameters with shape BxSx3x3,
              or None if build_intrinsics is False. Defined in pixels, with format:
              [[fx, 0, cx],
               [0, fy, cy],
               [0,  0,  1]]
              where fx, fy are focal lengths and (cx, cy) is the principal point,
              assumed to be at the center of the image (W/2, H/2).
    """

    intrinsics = None

    if pose_encoding_type == "absT_quaR_FoV":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]

        R = quat_to_mat(quat)
        extrinsics = torch.cat([R, T[..., None]], dim=-1)

        if build_intrinsics:
            H, W = image_size_hw
            fy = (H / 2.0) / torch.tan(fov_h / 2.0)
            fx = (W / 2.0) / torch.tan(fov_w / 2.0)
            intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
            intrinsics[..., 0, 0] = fx
            intrinsics[..., 1, 1] = fy
            intrinsics[..., 0, 2] = W / 2
            intrinsics[..., 1, 2] = H / 2
            intrinsics[..., 2, 2] = 1.0  # Set the homogeneous coordinate to 1
    else:
        raise NotImplementedError

    return extrinsics, intrinsics


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: Ground truth rotation matrices
        rot_pred: Predicted rotation matrices
        batch_size: Batch size for reshaping the result
        eps: Small value to avoid numerical issues

    Returns:
        Rotation angle error in degrees
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: Ground truth translation vectors
        tvec_pred: Predicted translation vectors
        batch_size: Batch size for reshaping the result
        ambiguity: Whether to handle direction ambiguity

    Returns:
        Translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: Ground truth translation vectors
        t: Predicted translation vectors
        eps: Small value to avoid division by zero
        default_err: Default error value for invalid cases

    Returns:
        Angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t
