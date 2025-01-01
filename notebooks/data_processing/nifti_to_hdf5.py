import os
import numpy as np
import nibabel as nib
from nibabel.orientations import aff2axcodes
import matplotlib.pyplot as plt
import h5py
from skimage.transform import resize

def get_user_confirmation(prompt):
    """
    Get user confirmation with proper input validation.
    Returns:
        - True if user enters 'y', 'Y', or just presses Enter
        - False if user enters 'n' or 'N'
        - None if user enters invalid input after 3 attempts
    """
    attempts = 0
    max_attempts = 3
    
    while attempts < max_attempts:
        response = input(prompt).lower().strip()
        
        # Handle empty input (just Enter) as Yes
        if response == "":
            return True
            
        # Handle yes/no responses
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
            
        # Invalid input
        print(f"Invalid input. Please enter 'y' or 'n' (or just press Enter for yes). {max_attempts - attempts - 1} attempts remaining.")
        attempts += 1
    
    print("\nMax attempts reached. Stopping processing for safety.")
    return None

def load_nifti_file(file_path):
    """Load a NIFTI file and return its data and orientation info."""
    img = nib.load(file_path)
    data = img.get_fdata()
    orientation = aff2axcodes(img.affine)
    return data, orientation, img.affine

def view_three_planes(data, masks=None, title="", slice_indices=None):
    """View sagittal, coronal, and axial planes."""
    if slice_indices is None:
        x_mid = data.shape[0] // 2
        y_mid = data.shape[1] // 2
        z_mid = data.shape[2] // 2
        slice_indices = (x_mid, y_mid, z_mid)
    else:
        x_mid, y_mid, z_mid = slice_indices

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.suptitle(f'{title}\nShape: {data.shape}', y=1.02, fontsize=12)

    # Sagittal view
    axes[0].imshow(data[x_mid, :, :], cmap='gray')
    if masks is not None:
        for mask_idx, mask in enumerate(masks):
            color = ['red', 'blue', 'green', 'yellow'][mask_idx % 4]
            mask_overlay = np.ma.masked_where(mask[x_mid, :, :] < 0.5, mask[x_mid, :, :])
            axes[0].imshow(mask_overlay, alpha=0.3, cmap=plt.cm.colors.ListedColormap([color]))
    axes[0].set_title(f'Sagittal (x={x_mid})')
    axes[0].axis('off')

    # Coronal view
    axes[1].imshow(data[:, y_mid, :], cmap='gray')
    if masks is not None:
        for mask_idx, mask in enumerate(masks):
            color = ['red', 'blue', 'green', 'yellow'][mask_idx % 4]
            mask_overlay = np.ma.masked_where(mask[:, y_mid, :] < 0.5, mask[:, y_mid, :])
            axes[1].imshow(mask_overlay, alpha=0.3, cmap=plt.cm.colors.ListedColormap([color]))
    axes[1].set_title(f'Coronal (y={y_mid})')
    axes[1].axis('off')

    # Axial view
    axes[2].imshow(data[:, :, z_mid], cmap='gray')
    if masks is not None:
        for mask_idx, mask in enumerate(masks):
            color = ['red', 'blue', 'green', 'yellow'][mask_idx % 4]
            mask_overlay = np.ma.masked_where(mask[:, :, z_mid] < 0.5, mask[:, :, z_mid])
            axes[2].imshow(mask_overlay, alpha=0.3, cmap=plt.cm.colors.ListedColormap([color]))
    axes[2].set_title(f'Axial (z={z_mid})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.draw()
    plt.pause(1)  # Show plot for 1 second
    plt.close(fig)  # Close the figure automatically

def try_orientations(data, masks=None, subject_id=None):
    """Automatically select orientation based on subject ID."""
    reorientations = [
        ("Original", lambda x: x, (0,1,2)),
        ("Reorder: 1,0,2", lambda x: np.transpose(x, (1,0,2)), (1,0,2)),
        ("Reorder: 1,2,0", lambda x: np.transpose(x, (1,2,0)), (1,2,0)),
        ("Reorder: 2,0,1", lambda x: np.transpose(x, (2,0,1)), (2,0,1)),
        ("Reorder: 2,1,0", lambda x: np.transpose(x, (2,1,0)), (2,1,0))
    ]

    # Extract subject number, handling cases with _V2 suffix
    if not subject_id:
        raise ValueError("Subject ID is required")
        
    try:
        # Split on '_' first to handle version suffixes
        base_subject = subject_id.split('_')[0]
        # Extract just the numeric portion
        raw_num = ''.join(filter(str.isdigit, base_subject))
        subject_num = int(raw_num)
        print(f"Subject ID: {subject_id}")
        print(f"Base subject: {base_subject}")
        print(f"Raw number extracted: {raw_num}")
        print(f"Final subject number: {subject_num}")
    except Exception as e:
        raise ValueError(f"Error extracting subject number from {subject_id}: {e}")
    
    # Select orientation based on subject number
    if subject_num < 94:  # Changed from <= 93 to < 94 for clarity
        choice = 3  # Use orientation 3 for subjects up to SBT093
        print(f"Subject {subject_num} < 94 (up to SBT093), using orientation 3")
    else:
        choice = 1  # Use orientation 1 for subjects from SBT094 onwards
        print(f"Subject {subject_num} >= 94 (SBT094 and above), using orientation 1")
    
    # Display the chosen orientation
    print(f"\nAutomatically selected orientation {choice} for subject {subject_id}")
    name, transform_func, axes_order = reorientations[choice-1]
    
    # Show the orientation
    transformed_data = transform_func(data)
    transformed_masks = [transform_func(mask) for mask in masks] if masks else None
    view_three_planes(transformed_data, transformed_masks, 
                     f"Selected {name}\nAxes order: {axes_order}")
    
    return transform_func, axes_order

def normalize_data(data):
    """Normalize data to [0, 1] range."""
    data = data - data.min()
    return data / (data.max() + 1e-8)

def resize_slice(data, target_size=(224, 224)):
    """Resize a 2D slice to target size."""
    return resize(data, target_size, mode='constant', anti_aliasing=True, preserve_range=True)

def process_to_h5(image_data, masks, output_path, subject_id):
    """Process and save data to HDF5 format. Skip slices without any mask data."""
    with h5py.File(output_path, 'a') as f:
        # Get the next available index
        existing_keys = [k for k in f.keys() if k.startswith('image_')]
        start_idx = len(existing_keys)
        curr_idx = start_idx

        # Process each slice
        num_slices = image_data.shape[2]
        for slice_idx in range(num_slices):
            # Check if the slice has all four masks
            has_all_masks = True
            for mask in masks:
                if not np.any(mask[:, :, slice_idx] > 0.5):  # Assuming mask threshold of 0.5
                    has_all_masks = False
                    break
            
            # Skip slice if it doesn't have all masks
            if not has_all_masks:
                if slice_idx % 10 == 0:
                    print(f"Skipping slice {slice_idx} - missing some masks")
                continue

            # Process image slice
            img_slice = image_data[:, :, slice_idx]
            img_slice = normalize_data(img_slice)
            img_slice = resize_slice(img_slice)
            img_slice = np.expand_dims(img_slice, axis=0)  # Add channel dimension

            # Process mask slices
            mask_slices = []
            for mask in masks:
                mask_slice = mask[:, :, slice_idx]
                mask_slice = resize_slice(mask_slice)
                mask_slices.append(mask_slice)
            mask_slices = np.stack(mask_slices, axis=0)

            # Save to HDF5
            f.create_dataset(f'image_{curr_idx}', data=img_slice)
            f.create_dataset(f'mask_{curr_idx}', data=mask_slices)

            if slice_idx % 10 == 0:
                print(f"Processed slice {slice_idx + 1}/{num_slices}")
            
            curr_idx += 1

        # Return the actual indices used (which may be fewer than the total number of slices)
        return start_idx, curr_idx - 1

def verify_h5_data(h5_path, start_idx, end_idx, step=100):
    """Verify the saved HDF5 data by displaying some slices."""
    with h5py.File(h5_path, 'r') as f:
        indices = range(start_idx, end_idx + 1, step)
        for idx in indices:
            image = f[f'image_{idx}'][()]
            masks = f[f'mask_{idx}'][()]
            
            plt.figure(figsize=(12, 4))
            
            # Show original image
            plt.subplot(121)
            plt.imshow(image[0], cmap='gray')
            plt.title(f'Image {idx}')
            plt.axis('off')
            
            # Show image with mask overlay
            plt.subplot(122)
            plt.imshow(image[0], cmap='gray')
            colors = ['red', 'blue', 'green', 'yellow']
            for mask_idx, mask in enumerate(masks):
                plt.imshow(np.ma.masked_where(mask < 0.5, mask),
                         alpha=0.3, cmap=plt.cm.colors.ListedColormap([colors[mask_idx]]))
            plt.title(f'Overlay {idx}')
            plt.axis('off')
            
            plt.tight_layout()
            plt.draw()
            plt.pause(1)  # Show plot for 1 second
            plt.close()  # Close the figure automatically

def main():
    # Paths
    base_dir = "/Volumes/advent/processed_data"
    h5_path = "/Volumes/advent/hdf5_processed_data/preprocessed_data.h5"
    
    # Get list of subjects
    subjects = sorted([d for d in os.listdir(base_dir) 
                      if os.path.isdir(os.path.join(base_dir, d))])
    
    for subject in subjects:
        print(f"\nProcessing {subject}")
        
        # Setup paths
        subject_dir = os.path.join(base_dir, subject)
        image_path = os.path.join(subject_dir, f"{subject}_images.nii")
        mask_paths = [
            os.path.join(subject_dir, f"{subject}_L_ES_II.nii"),
            os.path.join(subject_dir, f"{subject}_L_Mult_II.nii"),
            os.path.join(subject_dir, f"{subject}_R_ES_II.nii"),
            os.path.join(subject_dir, f"{subject}_R_Mult_II.nii")
        ]
        
        try:
            # Load and visualize original data
            print("Loading data...")
            image_data, orientation, _ = load_nifti_file(image_path)
            masks = []
            for mask_path in mask_paths:
                mask_data, _, _ = load_nifti_file(mask_path)
                masks.append(mask_data)
            
            print("Original data:")
            view_three_planes(image_data, masks, "Original Orientation")
            
            # Reorient if needed
            print("\nChecking orientation...")
            transform_func, axes_order = try_orientations(image_data, masks, subject)
            
            # Apply selected transformation
            image_data = transform_func(image_data)
            masks = [transform_func(mask) for mask in masks]
            
            # Verify orientation before processing
            print("\nVerifying orientation...")
            view_three_planes(image_data, masks, "Selected Orientation")
            
            # Get user confirmation to proceed
            proceed = get_user_confirmation("Proceed with processing? (y/n, or Enter for yes): ")
            if proceed is None:  # Invalid input after max attempts
                break
            if not proceed:
                print(f"Skipping processing for {subject}")
                continue
            
            # Process to HDF5
            print("\nProcessing to HDF5...")
            start_idx, end_idx = process_to_h5(image_data, masks, h5_path, subject)
            
            # Verify HDF5 data
            print("\nVerifying HDF5 data...")
            verify_h5_data(h5_path, start_idx, end_idx)
            
            # Get user confirmation to continue
            continue_processing = get_user_confirmation("\nContinue to next subject? (y/n, or Enter for yes): ")
            if continue_processing is None:  # Invalid input after max attempts
                break
            if not continue_processing:
                print("\nStopping processing as requested.")
                break
                
        except Exception as e:
            print(f"Error processing {subject}: {str(e)}")
            
            # Ask if user wants to continue after an error
            error_continue = get_user_confirmation("\nContinue to next subject despite error? (y/n, or Enter for yes): ")
            if error_continue is None or not error_continue:
                print("\nStopping processing as requested.")
                break
            continue

if __name__ == "__main__":
    main()
