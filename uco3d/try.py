from uco3d import UCO3DDataset, UCO3DFrameDataBuilder
from uco3d.dataset_utils.utils import get_dataset_root
from uco3d import opencv_cameras_projection_from_uco3d
import torch

# Get the dataset root folder and check that
# all required metadata files exist.
dataset_root = get_dataset_root(assert_exists=True)
# Get the "small" subset list containing a small subset
# of the uCO3D categories. For loading the whole dataset
# use "set_lists_all-categories.sqlite".

subset_lists_file = "/Users/aleksandrekandelaki/git/private/3d/uco3d/UCO3D_DEBUG/set_lists/set_lists_all-categories.sqlite"
dataset = UCO3DDataset(
    subset_lists_file=subset_lists_file,
    subsets=["train"],
    frame_data_builder=UCO3DFrameDataBuilder(
        apply_alignment=False,
        load_images=True,
        load_depths=False,
        load_masks=False,
        load_depth_masks=False,
        load_gaussian_splats=False,
        gaussian_splats_truncate_background=False,
        load_point_clouds=False,
        load_segmented_point_clouds=False,
        load_sparse_point_clouds=False,
        box_crop=False,
        box_crop_context=0.4,
        load_frames_from_videos=True,
        image_height=None,
        image_width=None,
        undistort_loaded_blobs=True,
    ),
    n_frames_per_sequence = 2
)
# query the dataset object to obtain a single video frame of a sequence

if __name__ == "__main__":
    # sequence_name = '475-99485-99578'

    for i in range(13):
        folder_name = "try"
        frame_data_1 = dataset[2*i]
        R_1, tvec_1, camera_matrix_1 = opencv_cameras_projection_from_uco3d(
            frame_data_1.camera,
            image_size=frame_data_1.image_size_hw[None],
        )  # R, tvec, camera_matrix follow OpenCV's camera definition

        frame_data_2 = dataset[2*i+1]
        R_2, tvec_2, camera_matrix_2 = opencv_cameras_projection_from_uco3d(
            frame_data_2.camera,
            image_size=frame_data_2.image_size_hw[None],
        )  # R, tvec, camera_matrix follow OpenCV's camera definition

        from PIL import Image

        # save bot images under "./images_to_infer/
        import os
        import numpy as np

        os.makedirs(f"./{folder_name}/{frame_data_1.sequence_name}/images_to_infer/", exist_ok=True)
        img_1 = frame_data_1.image_rgb.permute(1, 2, 0).numpy()
        # If float, convert to uint8
        if img_1.dtype != np.uint8:
            img_1 = (img_1 * 255).clip(0, 255).astype(np.uint8)

        img_2 = frame_data_2.image_rgb.permute(1, 2, 0).numpy()
        if img_2.dtype != np.uint8:
            img_2 = (img_2 * 255).clip(0, 255).astype(np.uint8)

        Image.fromarray(img_1).save(f"./{folder_name}/{frame_data_1.sequence_name}/images_to_infer/frame_0000.png")
        Image.fromarray(img_2).save(f"./{folder_name}/{frame_data_1.sequence_name}/images_to_infer/frame_0001.png")

        # # save camera parameters under "./cam_poses_gt/"
        os.makedirs(f"./{folder_name}/{frame_data_1.sequence_name}/cam_poses_gt/", exist_ok=True)
        torch.save({"R": R_1, "tvec": tvec_1, "camera_matrix": camera_matrix_1}, f"./{folder_name}/{frame_data_1.sequence_name}/cam_poses_gt/frame_0000.pt")
        torch.save({"R": R_2, "tvec": tvec_2, "camera_matrix": camera_matrix_2}, f"./{folder_name}/{frame_data_1.sequence_name}/cam_poses_gt/frame_0001.pt")

        # extri_opencv_frame_2 = torch.hstack(
    #     (R_1.squeeze(), tvec_1.T))  #make sure this is correct (take a look at pose_encoding_to_extri_intri)
    # extri_opencv_frame_3 = torch.hstack((R_50.squeeze(), tvec_50.T))
    #
    # gt_extri = torch.stack([extri_opencv_frame_2, extri_opencv_frame_3])
    #
    # predictions_mult_passes = []
    # for i in range(30):
    #     predictions_mult_passes.append(
    #         torch.load(f"/Users/aleksandrekandelaki/git/dl4sai/vggt/small_baseline_std_1/vggt_predictions_{i}.pt")["pose_enc"])
    #
    #     predictions = torch.cat(predictions_mult_passes, dim=0).mean(dim=0,keepdim=True)
    #     # R_1 is 1,3,3 shape and tvec_1 is 1,3. We need 3,4 extrinsic matrix in OpenCV format, which is [R|t] where R is 3x3 and t is 3x1.
    #
    #     extrinsic, _ = pose_encoding_to_extri_intri(predictions, build_intrinsics=False)
    #
    #     pred_extrinsic = extrinsic[0]
    #
    #     add_row = torch.tensor([0, 0, 0, 1]).expand(pred_extrinsic.size(0), 1, 4)
    #
    #     pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
    #     gt_se3 = torch.cat((gt_extri, add_row), dim=1)
    #
    #     rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, 2)
    #
    #     Racc_5 = (rel_rangle_deg < 5).float().mean().item()
    #     Tacc_5 = (rel_tangle_deg < 5).float().mean().item()
    #
    #     print(f"Pass {i}: R_ACC@5: {Racc_5:.4f}")
    #     print(f"Pass {i}: T_ACC@5: {Tacc_5:.4f}")
    #
    #     print(f"Pass {i}: relative rotation angle error (degrees): {rel_rangle_deg}")
    #     print(f"Pass {i}: relative translation angle error (degrees): {rel_tangle_deg}")
