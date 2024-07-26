import os
import random
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import nibabel as nib
import cv2
from argparse import ArgumentParser

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)


def parse_seg_args():
    """Parse arguments for segmentation tasks."""
    parser = ArgumentParser()
    base_dir = Path(__file__).resolve().parent

    parser.add_argument(
        '--data_root', type=str, default=os.path.join(base_dir, 'data', 'clean'),
        help='Root directory of dataset'
    )
    parser.add_argument(
        '--cases_split', type=str, default=os.path.join(base_dir, 'data', 'brats2021_split.csv'),
        help='CSV file containing the case splits'
    )
    parser.add_argument(
        '--meta_cases_split', type=str, default=os.path.join(base_dir, 'data', 'split', 'brats2021_split.csv'),
        help='CSV file containing the case splits'
    )
    parser.add_argument(
        '--noise_data_root', type=str, default=os.path.join(base_dir, 'data', 'train_data'),
        help='Root directory for noisy data'
    )

    # Noise arguments
    parser.add_argument(
        '--noise_type', type=str, default='erosion', choices=['erosion', 'dilation', 'mixed'],
        help='Type of noise to apply'
    )
    parser.add_argument(
        '--size_kernel', type=int, default=7,
        help='Size of the kernel for noise application'
    )
    parser.add_argument(
        '--corruption_rate', type=int, default=100, choices=[0, 20, 40, 60, 80, 100],
        help='Percentage of metadata to be corrupted'
    )

    return parser.parse_args()


def apply_noise(image_path, output_path, noise_type, size_kernel):
    """Apply specified noise to a NIfTI image."""
    nifti_img = nib.load(image_path)
    image_data = nifti_img.get_fdata()
    kernel = np.ones((size_kernel, size_kernel))

    if noise_type == 'erosion':
        noised_image_data = cv2.erode(image_data, kernel)
    elif noise_type == 'dilation':
        noised_image_data = cv2.dilate(image_data, kernel)
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    # Save the processed image
    noised_nifti_img = nib.Nifti1Image(noised_image_data, nifti_img.affine, nifti_img.header)
    nib.save(noised_nifti_img, output_path)


def process_cases(input_directory, output_directory, meta_train_cases, selected_cases, all_data, noise_type,
                  size_kernel, corruption_rate):
    """
    Processes cases by applying noise and copying modality files.

    Parameters:
    - input_directory: Directory containing the original images.
    - output_directory: Directory where processed images will be saved.
    - meta_train_cases: List of cases to be processed.
    - noise_type: Type of noise to apply.
    - size_kernel: Size of the kernel for noise application.
    - corruption_rate: Percentage of metadata to be corrupted.
    """
    output_directory = Path(output_directory) / 'brats2021'
    input_directory = Path(input_directory) / 'brats2021'

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)

    os.makedirs(output_directory, exist_ok=True)
    print(f"Created output directory: {output_directory}")

    all_data = [case for case in all_data]
    meta_train = [case for case in meta_train_cases if case not in selected_cases]
    total_cases = len(meta_train_cases)
    num_cases_to_process = int((corruption_rate / 100) * total_cases)
    print(num_cases_to_process)
    print(f"Processing {num_cases_to_process} out of {total_cases} total_cases cases.")

    if num_cases_to_process > 0:
        cases_to_process = np.random.choice(meta_train, num_cases_to_process, replace=False)
    else:
        cases_to_process = []

    print(f"Selected cases to process: {cases_to_process}")

    if noise_type == 'mixed':
        num_erosion_cases = num_cases_to_process // 2
        num_dilation_cases = num_cases_to_process - num_erosion_cases

        erosion_cases = np.random.choice(cases_to_process, num_erosion_cases, replace=False)


        remaining_cases = list(set(cases_to_process) - set(erosion_cases))

        dilation_cases = np.random.choice(remaining_cases, num_dilation_cases, replace=False)



        print(f"Erosion cases: {erosion_cases}")
        print(f"Dilation cases: {dilation_cases}")

    for case in all_data:
        case_path = os.path.join(input_directory, case)
        output_case_path = os.path.join(output_directory, case)
        print(output_case_path)
        os.makedirs(output_case_path, exist_ok=True)

        files = os.listdir(case_path)

        if case in cases_to_process:
            if noise_type == 'mixed':
                current_noise_type = 'erosion' if case in erosion_cases else 'dilation'
            else:
                current_noise_type = noise_type

            for file in files:
                if '_seg' in file and file.endswith('.nii.gz'):
                    label_path = os.path.join(case_path, file)
                    output_label_path = os.path.join(output_case_path, file)
                    apply_noise(label_path, output_label_path, current_noise_type, size_kernel)
                elif any(x in file for x in ['_t1', '_t2', '_flair', '_t1c']) and file.endswith('.nii.gz'):
                    original_file_path = os.path.join(case_path, file)
                    output_file_path = os.path.join(output_case_path, file)
                    shutil.copy(original_file_path, output_file_path)
        else:
            for file in files:
                original_file_path = os.path.join(case_path, file)
                output_file_path = os.path.join(output_case_path, file)
                shutil.copy(original_file_path, output_file_path)

    return sorted(cases_to_process, key=lambda x: meta_train_cases.index(x))


def select_and_update_cases(original_csv_file_path, updated_csv_file_path, num_cases=10, seed=42):
    """Select examples from the training data and add new rows labeled as meta_train in a new CSV file."""
    if os.path.exists(updated_csv_file_path):
        os.remove(updated_csv_file_path)

    df = pd.read_csv(original_csv_file_path)
    train_cases = df[df['split'] == 'train']['name'].tolist()

    random.seed(seed)

    if args.corruption_rate < 100:
        random.seed(seed)
        selected_cases = random.sample(train_cases, num_cases)

        meta_train_df = pd.DataFrame({
            'name': selected_cases,
            'split': ['meta_train'] * num_cases
        })

        updated_df = pd.concat([df, meta_train_df], ignore_index=True)
    else:
        selected_cases = []
        updated_df = df

    updated_df.to_csv(updated_csv_file_path, index=False)
    print(f"Selected cases for meta_train: {selected_cases}")
    print(f"Added {len(selected_cases)} new rows labeled as meta_train in the new CSV file.")
    return selected_cases

def main():


    # Update CSV file with selected meta_train cases and save as a new file
    selected_meta_train_cases = select_and_update_cases(args.cases_split, args.meta_cases_split)

    df = pd.read_csv(args.cases_split)
    all_data = df['name'].tolist()
    meta_train = df[df['split'] == 'train']['name'].tolist()

    meta_train_cases = pd.read_csv(args.meta_cases_split)[pd.read_csv(args.meta_cases_split)['split'] == 'meta_train'][
        'name'].tolist()
    cases_to_process = process_cases(
        args.data_root, args.noise_data_root, meta_train,
        selected_meta_train_cases, all_data, args.noise_type, args.size_kernel, args.corruption_rate
    )

    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    args = parse_seg_args()
    main()
