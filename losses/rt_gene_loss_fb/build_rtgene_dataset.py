# #!/usr/bin/env fbpython
# # (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# # NOTE this code is for RESEARCH USE ONLY - NO PRODUCTION USE

# from __future__ import absolute_import, division, print_function

# import argparse
# import os
# from itertools import repeat
# from typing import List, Optional, Tuple

# import h5py
# import numpy as np
# from bc.utils.data.path_manager import get_path_manager
# from iopath.common.file_io import PathManager
# from PIL import Image, ImageFilter, ImageOps
# from torchvision import transforms
# from tqdm import tqdm


# # pyre-ignore
# def get_rt_gene_augmentation(_required_size: tuple[int, int]):
#     # Augmentations following `prepare_dataset.m`: randomly crop and resize the image 10 times,
#     # along side two blurring stages, grayscaling and histogram normalisation
#     # NOTE this is on purpose a list and not a compose() - see usage below
#     _transforms_list = [
#         transforms.RandomResizedCrop(
#             size=_required_size, scale=(0.85, 1.0)
#         ),  # equivalent to random 5px from each edge
#         transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
#         transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
#         transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
#         transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
#         transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
#         transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
#         transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
#         transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
#         transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
#         transforms.Grayscale(num_output_channels=3),
#         lambda x: x.filter(ImageFilter.GaussianBlur(radius=1)),
#         lambda x: x.filter(ImageFilter.GaussianBlur(radius=3)),
#         lambda x: ImageOps.equalize(x),
#     ]  # histogram equalisation
#     return _transforms_list


# def load_and_augment(
#     file_path: str, required_size: tuple[int, int], augment: bool = False
# ) -> np.ndarray:
#     image = Image.open(file_path).resize(required_size)
#     augmented_images = [
#         np.array(trans(image))
#         for trans in get_rt_gene_augmentation(required_size)
#         if augment is True
#     ]
#     augmented_images.append(np.array(image))

#     return np.array(augmented_images, dtype=np.uint8)


# import concurrent.futures


# def process_line(
#     line: str, subject_data: str
# ) -> Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
#     split = line.split(",")
#     image_name = "{:0=6d}".format(int(split[0]))

#     try:
#         left_img_path = path_manager.get_local_path(
#             os.path.join(
#                 subject_data,
#                 "inpainted/left/",
#                 "left_{:0=6d}_rgb.png".format(int(split[0])),
#             )
#         )
#         right_img_path = path_manager.get_local_path(
#             os.path.join(
#                 subject_data,
#                 "inpainted/right/",
#                 "right_{:0=6d}_rgb.png".format(int(split[0])),
#             )
#         )

#         head_phi = float(split[1].strip()[1:])
#         head_theta = float(split[2].strip()[:-1])
#         gaze_phi = float(split[3].strip()[1:])
#         gaze_theta = float(split[4].strip()[:-1])
#         labels = np.array([(head_theta, head_phi), (gaze_theta, gaze_phi)])

#         left_data = load_and_augment(
#             left_img_path, _required_size, augment=args.augment_dataset
#         )
#         right_data = load_and_augment(
#             right_img_path, _required_size, augment=args.augment_dataset
#         )

#         return image_name, left_data, right_data, labels
#     except FileNotFoundError:
#         return None


# def parse_dataset(args: argparse.Namespace) -> None:
#     _compression: Optional[str] = "lzf" if args.compress is True else None

#     subject_path: List[str] = [
#         os.path.join(args.rt_gene_root, "s{:03d}_glasses/".format(_i))
#         for _i in range(0, 17)
#     ]

#     hdf_file = h5py.File(
#         os.path.abspath(os.path.join("/tmp", "rtgene_dataset.hdf5")),
#         mode="w",
#     )
#     not_found = 0
#     for subject_id, subject_data in enumerate(subject_path):
#         subject_id = str("s{:03d}".format(subject_id))
#         subject_grp = hdf_file.create_group(subject_id)
#         local_path = path_manager.get_local_path(
#             os.path.join(subject_data, "label_combined.txt")
#         )
#         print(f"Working on {subject_id}")
#         with open(local_path, "r") as f:
#             _lines = f.readlines()

#             with concurrent.futures.ThreadPoolExecutor(max_workers=48) as executor:
#                 results = list(
#                     tqdm(
#                         executor.map(process_line, _lines, repeat(subject_data)),
#                         total=len(_lines),
#                     )
#                 )

#         print("Generating dataset")
#         for result in results:
#             if result is not None:
#                 image_name, left_data, right_data, labels = result

#                 image_grp = subject_grp.create_group(image_name)
#                 image_grp.create_dataset(
#                     "left", data=left_data, compression=_compression
#                 )
#                 image_grp.create_dataset(
#                     "right", data=right_data, compression=_compression
#                 )
#                 image_grp.create_dataset("label", data=labels)
#             else:
#                 not_found += 1
#     print(f"Skipped {not_found}")

#     hdf_file.flush()
#     hdf_file.close()


# if __name__ == "__main__":
#     _required_size = (224, 224)

#     parser = argparse.ArgumentParser(description="Build RT-GENE dataset h5 files")
#     # parser.add_argument("--bucket", type=str, required=True, help="Manifold bucket")
#     parser.add_argument(
#         "--rt_gene_root",
#         type=str,
#         required=True,
#         nargs="?",
#         help="Path to the base directory of RT_GENE",
#     )
#     parser.add_argument(
#         "--augment_dataset",
#         type=bool,
#         required=False,
#         default=False,
#         help="Whether to augment the dataset with predefined transforms",
#     )
#     parser.add_argument("--compress", action="store_true", dest="compress")
#     parser.add_argument("--no-compress", action="store_false", dest="compress")
#     parser.set_defaults(compress=False)
#     # pyre-ignore
#     args = parser.parse_args()
#     path_manager: PathManager = get_path_manager()
#     parse_dataset(args)
