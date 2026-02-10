from uco3d import UCO3DDataset, UCO3DFrameDataBuilder
from uco3d.dataset_utils.utils import get_dataset_root
from uco3d import opencv_cameras_projection_from_uco3d
import os
# Get the dataset root folder and check that
# all required metadata files exist.
dataset_root = get_dataset_root(assert_exists=True)
# Get the "small" subset list containing a small subset
# of the uCO3D categories. For loading the whole dataset
# use "set_lists_all-categories.sqlite".
# subset_lists_file = os.path.join(
#     dataset_root,
#     "set_lists", 
#     "set_lists_3categories-debug.sqlite",
# )

subset_lists_file = "/home/stud/kandelak/git/uco3d/set_lists_3categories-debug.sqlite"
dataset = UCO3DDataset(
    subset_lists_file=subset_lists_file,
    subsets=["train"],
    frame_data_builder=UCO3DFrameDataBuilder(
        apply_alignment=False,
        load_images=False,
        load_depths=False,
        load_masks=False,
        load_depth_masks=False,
        load_gaussian_splats=False,
        gaussian_splats_truncate_background=False,
        load_point_clouds=False,
        load_segmented_point_clouds=False,
        load_sparse_point_clouds=False,
        box_crop=True,
        box_crop_context=0.4,
        load_frames_from_videos=True,
        image_height=800,
        image_width=800,
        undistort_loaded_blobs=True,
    )
)
# query the dataset object to obtain a single video frame of a sequence

frame_data = dataset[100]
R, tvec, camera_matrix = opencv_cameras_projection_from_uco3d(
    frame_data.camera,
    image_size=frame_data.image_size_hw[None],
)  # R, tvec, camera_matrix follow OpenCV's camera definition

print(R)