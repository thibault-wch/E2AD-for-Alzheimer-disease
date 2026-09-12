import os
import pickle
import torch
import torch.nn.functional as F
import monai.transforms as mtransforms


def mask_to_one_hot(mask):
    """Converts a mask to one-hot encoding, dropping the background class (index 0)."""
    mask = mask.squeeze(0).type(torch.long)
    one_hot_mask = F.one_hot(mask, num_classes=57)
    # Permute to (C, D, H, W) and drop background class
    one_hot_mask = one_hot_mask.permute(3, 0, 1, 2)[1:]
    return one_hot_mask.float()


def pair_form(input_dict, target_key):
    """Separates the target fold for testing and aggregates the remaining for training."""
    target_list = input_dict.get(target_key, [])
    non_target_lists = []

    for key, value in input_dict.items():
        if key != target_key:
            non_target_lists.extend(value)

    return target_list, non_target_lists


def extract_subgroups(pairs, mode='NCAD'):
    """Dynamically extracts specific subgroup labels across all folds."""
    valid_labels = ('0', '1') if mode == 'NCAD' else ('3', '4')

    return {
        fold_idx: [item for item in fold_data if item[1] in valid_labels]
        for fold_idx, fold_data in pairs.items()
    }


class OurDataset(torch.utils.data.Dataset):
    """
    Multimodal Neuroimaging Dataset

    Parameters:
        root (int/str) : Identifier for root directory path.
        mtype (str)    : Classification task type ('NCAD' or 'SPMCI').
        mode (str)     : 'train' or 'test'.
        type (str)     : Modality type ('single', 'multi_mri', 'multi_pet').
        fold (int)     : Current cross-validation fold.
    """

    def __init__(self, root=1, mtype='NCAD', mode="train", type="single", fold=0):
        self.mode = mode
        self.type = type

        # Setup base directory
        if root == 1:
            self.basic_dir = '/data/chwang/AtlasAAAI/'
        else:
            self.basic_dir = '/mnt/miah203/chwang/AtlasProject/data/'

        # Identify correct dictionary based on modality
        dict_mapping = {
            'single': 'all_mri_dict.pkl',
            'multi_mri': 'pure_mri_dict.pkl',
            'multi_pet': 'paired_dict.pkl'
        }

        if type not in dict_mapping:
            raise ValueError(f"Unsupported modality type: {type}")

        # Safely load data dictionary
        dict_path = os.path.join(self.basic_dir, dict_mapping[type])
        with open(dict_path, 'rb') as f:
            self.data_pairs = pickle.load(f)

        # Organize pairs for current fold
        self.organized_pairs = extract_subgroups(self.data_pairs, mtype)
        self.test_pairs, self.train_pairs = pair_form(self.organized_pairs, fold)

        # ---------------------------------------------------------
        # Data Preprocessing Pipelines
        # ---------------------------------------------------------

        # Shared pipeline elements for standard imaging
        shared_intensity_crop = [
            mtransforms.ScaleIntensityRangePercentiles(
                lower=0, upper=99, b_min=-1.0, b_max=1.0, clip=True, relative=False
            ),
            mtransforms.SpatialCrop(roi_center=(128, 128, 128), roi_size=(192, 224, 192))
        ]

        self.basic_mri_transform = mtransforms.Compose([
            mtransforms.LoadImage(image_only=True),
            mtransforms.EnsureChannelFirst(),
            mtransforms.SqueezeDim(),
            mtransforms.EnsureChannelFirst(),
            mtransforms.EnsureType(),
            *shared_intensity_crop
        ])

        self.basic_pet_transform = mtransforms.Compose([
            mtransforms.LoadImage(image_only=True),
            mtransforms.EnsureChannelFirst(),
            mtransforms.SqueezeDim(),
            mtransforms.EnsureChannelFirst(),
            mtransforms.EnsureType(),
            *shared_intensity_crop
        ])

        self.basic_atlas_transform = mtransforms.Compose([
            mtransforms.LoadImage(image_only=True),
            mtransforms.EnsureChannelFirst(),
            mtransforms.SqueezeDim(),
            mtransforms.EnsureChannelFirst(),
            mtransforms.EnsureType(),
            mtransforms.SpatialCrop(roi_center=(128, 128, 128), roi_size=(192, 224, 192)),
            mtransforms.Resize(spatial_size=(24, 28, 24), mode='nearest'),
            mtransforms.Flip(spatial_axis=-3)
        ])

        # ---------------------------------------------------------
        # Data Formulation
        # ---------------------------------------------------------
        if mode == "train":
            ratio = 1.0
            half_ratio = ratio / 2
            split_idx_1 = int(half_ratio * len(self.train_pairs))
            split_idx_2 = int((1 - half_ratio) * len(self.train_pairs))
            self.imgs = self.train_pairs[:split_idx_1] + self.train_pairs[split_idx_2:]
        elif mode == "test":
            self.imgs = self.test_pairs

        # Fast lookup mapping for string labels to numeric and directory targets
        self.label_map = {
            '0': (0, 'NC'),
            '1': (1, 'AD'),
            '3': (0, 'SMCI'),
            '4': (1, 'PMCI')
        }

    def __getitem__(self, index):
        # Retrieve subject ID and target label
        subject_id, str_label = self.imgs[index][0], self.imgs[index][1]
        label, tlabel = self.label_map[str_label]

        # Construct paths
        mri_path = os.path.join(self.basic_dir, 'MRI', tlabel, f"{subject_id}.nii.gz")
        atlas_path = os.path.join(self.basic_dir, 'flirt_atlas', tlabel, f"{subject_id}.nii.gz")

        # Process core images
        A = self.basic_mri_transform(mri_path)
        atlas = self.basic_atlas_transform(atlas_path)
        atlas_one_hot = mask_to_one_hot(atlas)

        # Output payload
        if self.type == 'multi_pet':
            pet_path = os.path.join(self.basic_dir, 'Final_PET', tlabel, f"{subject_id}.nii.gz")
            B = self.basic_pet_transform(pet_path)

            return {
                'idx_lb': subject_id,
                'mri_x_lb': A,
                'pet_x_lb': B,
                'atlas_x_lb': atlas_one_hot,
                'y_lb': label,
            }
        else:
            return {
                'idx_lb': subject_id,
                'x_lb': A,
                'atlas_x_lb': atlas_one_hot,
                'y_lb': label
            }

    def __len__(self):
        return len(self.imgs)