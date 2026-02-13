import os
import torch
import matplotlib.pyplot as plt

from pose_enc_utils import pose_encoding_to_extri_intri
from test_co3d import se3_to_relative_pose_error
from uco3d import UCO3DDataset, UCO3DFrameDataBuilder
from uco3d.dataset_utils.utils import get_dataset_root
from uco3d import opencv_cameras_projection_from_uco3d

if __name__ == "__main__":
    # Dataset setup (same as yours)
    dataset_root = get_dataset_root(assert_exists=True)

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
            image_height=1920,
            image_width=1080,
            undistort_loaded_blobs=True,
        ),
    )
    sequences = ['1-89973-52840', '1-90320-26441', '1-95979-13606', '10-81738-74119', '117-3686-24815',
                 '12-10308-28681', '13-68075-18110', '156-6699-92799', '16-11679-97735', '20-38390-68010',
                 '2017-26879-34905', '21-1781-33291', '26-14460-30571', '267-59948-42658', '27-79013-34290',
                 '27-92237-65486', '28006-14019-9638', '3-9266-65507', '305-68822-66727', '33-11430-17759',
                 '38-75852-49792', '4-91634-39388', '4-91751-93241', '401-36913-71936', '411-19840-28276',
                 '42033-32228-40100', '47433-58842-22217', '47480-57073-46870', '475-99485-99578', '494-29795-17364',
                 '533-16958-69462', '538-38847-906', '54207-48862-40002', '57-41030-21122', '60458-35176-64843',
                 '605-31775-92635', '62387-12826-25112', '65154-22082-61695', '654-18524-38608', '69-40182-31015',
                 '7-88671-19551', '70-1164-44120', '732-27981-57173', '767-14445-38072', '8-20919-19870']

    for sequence in sequences:
        # sequence = '1-89973-52840'
        # # Load frames and GT camera params
        # frame_data_1 = dataset[("57-41030-21122", 0)]
        # R_1, tvec_1, camera_matrix_1 = opencv_cameras_projection_from_uco3d(
        #     frame_data_1.camera,
        #     image_size=frame_data_1.image_size_hw[None],
        # )
        #
        # frame_data_2 = dataset[("57-41030-21122", 50)]
        # R_2, tvec_2, camera_matrix_50 = opencv_cameras_projection_from_uco3d(
        #     frame_data_2.camera,
        #     image_size=frame_data_2.image_size_hw[None],
        # )

        cam_frame_1 = torch.load(
            f"/Users/aleksandrekandelaki/git/private/3d/uco3d/uco3d/random_2_view_samples/{sequence}/cam_poses_gt/frame_0000.pt")
        cam_frame_2 = torch.load(
            f"/Users/aleksandrekandelaki/git/private/3d/uco3d/uco3d/random_2_view_samples/{sequence}/cam_poses_gt/frame_0001.pt")

        R_1 = cam_frame_1["R"]
        tvec_1 = cam_frame_1["tvec"]
        R_2 = cam_frame_2["R"]
        tvec_2 = cam_frame_2["tvec"]
        # Build GT se3 (same approach as yours; keep if it's correct for your pipeline)
        extri_opencv_frame_2 = torch.hstack((R_1.squeeze(), tvec_1.T))
        extri_opencv_frame_3 = torch.hstack((R_2.squeeze(), tvec_2.T))
        gt_extri = torch.stack([extri_opencv_frame_2, extri_opencv_frame_3])

        # --- Accumulate predictions and measure effect of #samples ---
        predictions_mult_passes = []
        samples = []
        rel_r_errs = []
        rel_t_errs = []

        num_total_passes = 20

        for i in range(num_total_passes):
            predictions_mult_passes.append(
                torch.load(
                    f"/Users/aleksandrekandelaki/git/dl4sai/vggt/small_baseline_std_1/{sequence}/vggt_predictions_{i}.pt")[
                    "pose_enc"])

            n = i + 1
            predictions = torch.cat(predictions_mult_passes, dim=0).mean(dim=0, keepdim=True)

            extrinsic, _ = pose_encoding_to_extri_intri(predictions, build_intrinsics=False)

            pred_extrinsic = extrinsic[0]

            add_row = torch.tensor([0, 0, 0, 1]).expand(pred_extrinsic.size(0), 1, 4)

            pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
            gt_se3 = torch.cat((gt_extri, add_row), dim=1)

            rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, 2)

            print(f"Pass {n}: rel_rangle_deg = {rel_rangle_deg.item():.4f}, rel_tangle_deg = {rel_tangle_deg.item():.4f}")
            samples.append(n)
            rel_r_errs.append(rel_rangle_deg.item())
            rel_t_errs.append(rel_tangle_deg.item())

        # --- Plot: errors on left y-axis, variance on right y-axis ---
        fig, ax1 = plt.subplots(figsize=(9, 5))

        ax1.plot(samples, rel_r_errs, marker="o", label="Rotation")
        ax1.plot(samples, rel_t_errs, marker="o", label="Translation")
        ax1.set_xlabel("Number of samples")
        ax1.set_ylabel("Error (deg)")
        ax1.grid(True, alpha=0.3)

        # plot legend for the left y-axis
        ax1.legend(loc="upper right")

        plt.title("Effect of #Samples on Camera Pose Error")
        plt.tight_layout()

        plt.savefig(os.path.join(f"UCO3D_DEBUG_ANALYSIS/{sequence}_pose_error_and_variance_vs_samples.png"))
