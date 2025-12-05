"""Edit video using python

Usage:
    python3 ./src/edit_video.py [attack_type]
    
Supported attack types:
    - basic_edit
    - scaling_attack (default)
    - region_replace 
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from utils.video import load_video, write_video, export_frame, load_image


def main():
    Y_VIDEO_FILE = 'in/irl/c2/iphone/38.MOV'
    
    # Default attack type
    attack_type = 'scaling_attack'
    
    # Check if user provided attack type argument
    if len(sys.argv) > 1:
        attack_type = sys.argv[1].lower()
    
    print(f'Loading video file {Y_VIDEO_FILE}')
    y, VIDEO_FPS = load_video(Y_VIDEO_FILE, downscale_factor=4, gamma_correction=2.2)
    export_frame(y, 0, 'out/true_y_frame_0.png')

    # Select attack based on user input or default
    if attack_type == 'basic_edit':
        print('Applying basic_edit attack...')
        y_edited = basic_edit(y, reference=load_image('in/irl/c2/iphone/fake_y_frame_0.png'))
        output_file = 'out/fake_y_basic_edit.mp4'
    
    elif attack_type == 'scaling_attack':
        print('Applying scaling_attack...')
        y_edited = scaling_attack(y, reference=load_image('in/irl/c2/iphone/fake_y_frame_0.png'))
        output_file = 'out/fake_y_scaling.mp4'
    
    elif attack_type == 'region_replace':
        print('Applying region_replace_attack...')
        # Default region replace parameters
        source_time = (0, 2)           # First 2 seconds as source
        source_roi = (100, 100, 200, 200)  # 100x100 pixel region
        target_time = (5, 7)           # Replace frames 5-7 seconds
        target_roi = (300, 300, 400, 400)  # Target region
        
        y_edited = region_replace_attack(
            y,
            source_time=source_time,
            source_roi=source_roi,
            target_time=target_time,
            target_roi=target_roi,
            fps=VIDEO_FPS,
            swap=False
        )
        output_file = 'out/fake_y_region_replace.mp4'
    
    else:
        print(f'Unknown attack type: {attack_type}')
        print('Supported types: basic_edit, scaling_attack, region_replace')
        print('Using default (scaling_attack)...')
        y_edited = scaling_attack(y, reference=load_image('in/irl/c2/iphone/fake_y_frame_0.png'))
        output_file = 'out/fake_y_scaling.mp4'

    # export edited video
    write_video(y_edited, output_file, VIDEO_FPS, gamma=2.2)
    print(f'Video saved to {output_file}')


def basic_edit(y, reference=None):
    if reference is None:
        # Simulate malicious video cut
        y_edited = np.concatenate((y[:y.shape[0]//4], y[2*y.shape[0]//4:]))
        # Simulate malicious photoshop (edit in a white square in the middle)
        y_edited[0:y_edited.shape[0]//8, y_edited.shape[1]//4:y_edited.shape[1]//2, y_edited.shape[2]//4:y_edited.shape[2]//2] = 1
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            reference_uint8 = (255*reference).astype(np.uint8)
            y_uint8 = (255*y[0]).astype(np.uint8)
            modified_pixels = np.any(np.abs(reference_uint8-y_uint8) > 0, axis=2)
            print(f'Modified {np.count_nonzero(modified_pixels)} pixels')
            # debug
            print(f'{np.transpose(modified_pixels.nonzero())}')
            y_edited = np.copy(y)
            y_edited[:, modified_pixels] = reference[modified_pixels]

    return y_edited


def scaling_attack(y, reference=None):
    if reference is None:
        y_edited = y
        # Multiply top left quadrant by random alpha
        region = y_edited[:, :y_edited.shape[1]//2, :y_edited.shape[2]//2]
        alpha = np.random.uniform(0/np.max(region), 1/np.max(region), (region.shape[1], region.shape[2], 3))
        y_edited[:, :y_edited.shape[1]//2, :y_edited.shape[2]//2] = \
            alpha*region
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            reference_uint8 = (255*reference).astype(np.uint8)
            y_uint8 = (255*y[0]).astype(np.uint8)
            mask = np.where(y_uint8==0, 1, reference_uint8/y_uint8)
            modified_pixels = np.any(mask!=1, axis=2)
            print(f'Modified {np.count_nonzero(modified_pixels)} pixels')
            # debug
            print(f'{np.transpose(modified_pixels.nonzero())}')
            y_edited = mask*y
    
    return y_edited


def region_replace_attack(frames, source_time, source_roi, target_time, target_roi, fps, swap=False):
    """
    Wrapper function to apply region replace attack to video frames.
    
    Args:
        frames: Video frames array (T, H, W, C)
        source_time: Tuple (start_sec, end_sec) for source region
        source_roi: Tuple (x1, y1, x2, y2) for source spatial region
        target_time: Tuple (start_sec, end_sec) for target frames
        target_roi: Tuple (x1, y1, x2, y2) for target spatial region
        fps: Frames per second
        swap: If True, swap regions; if False, just replace
    
    Returns:
        Modified frames array
    """
    return replace_region(frames, source_time, source_roi, target_time, target_roi, fps, swap)


# ============================================================================
# REGION REPLACE ATTACK FUNCTIONS (from region_replace.py)
# ============================================================================


def extract_first_frame(frames, output_path):
    """Extract the first frame from video and save it as an image."""
    if frames.shape[0] > 0:
        first_frame = frames[0]
        
        # Convert RGB to BGR for cv2.imwrite
        first_frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the frame
        cv2.imwrite(output_path, first_frame_bgr)
        print(f"Extracted first frame and saved to {output_path}")
    else:
        print("Error: No frames available to extract")


def visualize_regions(frames, source_roi, target_roi, output_path):
    """
    Visualize source and target regions on the first frame.
    Helps users verify they have the correct coordinates.
    """
    if frames.shape[0] == 0:
        print("Error: No frames available")
        return
    
    first_frame = frames[0].copy()
    
    source_x1, source_y1, source_x2, source_y2 = source_roi
    target_x1, target_y1, target_x2, target_y2 = target_roi
    
    # Draw rectangles on the frame
    # Source region in green
    cv2.rectangle(first_frame, (source_x1, source_y1), (source_x2, source_y2), (0, 255, 0), 3)
    cv2.putText(first_frame, "SOURCE", (source_x1, source_y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Target region in red
    cv2.rectangle(first_frame, (target_x1, target_y1), (target_x2, target_y2), (255, 0, 0), 3)
    cv2.putText(first_frame, "TARGET", (target_x1, target_y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save visualization
    first_frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, first_frame_bgr)
    print(f"Saved region visualization to {output_path}")


def replace_region(frames, source_time, source_roi, target_time, target_roi, fps, swap=False):
    """
    Replace or swap regions in target frames with/from source frames.
    
    Args:
        frames: Video frames array (T, H, W, C)
        source_time: Tuple (start_sec, end_sec) for source region
        source_roi: Tuple (x1, y1, x2, y2) for source spatial region
        target_time: Tuple (start_sec, end_sec) for target frames
        target_roi: Tuple (x1, y1, x2, y2) for target spatial region
        fps: Frames per second
        swap: If True, swap regions; if False, just replace
    
    Returns:
        Modified frames array
    """
    frames_modified = frames.copy()
    
    # Convert time to frame indices
    source_start_frame = int(source_time[0] * fps)
    source_end_frame = int(source_time[1] * fps)
    target_start_frame = int(target_time[0] * fps)
    target_end_frame = int(target_time[1] * fps)
    
    # Validate frame indices
    source_start_frame = max(0, min(source_start_frame, frames.shape[0] - 1))
    source_end_frame = max(0, min(source_end_frame, frames.shape[0]))
    target_start_frame = max(0, min(target_start_frame, frames.shape[0] - 1))
    target_end_frame = max(0, min(target_end_frame, frames.shape[0]))
    
    # Parse ROI coordinates
    source_x1, source_y1, source_x2, source_y2 = source_roi
    target_x1, target_y1, target_x2, target_y2 = target_roi
    
    # Ensure x1 < x2 and y1 < y2
    source_x1, source_x2 = min(source_x1, source_x2), max(source_x1, source_x2)
    source_y1, source_y2 = min(source_y1, source_y2), max(source_y1, source_y2)
    target_x1, target_x2 = min(target_x1, target_x2), max(target_x1, target_x2)
    target_y1, target_y2 = min(target_y1, target_y2), max(target_y1, target_y2)
    
    # Clamp to image boundaries
    source_x1 = max(0, source_x1)
    source_y1 = max(0, source_y1)
    source_x2 = min(frames.shape[2], source_x2)
    source_y2 = min(frames.shape[1], source_y2)
    
    target_x1 = max(0, target_x1)
    target_y1 = max(0, target_y1)
    target_x2 = min(frames.shape[2], target_x2)
    target_y2 = min(frames.shape[1], target_y2)
    
    source_height = source_y2 - source_y1
    source_width = source_x2 - source_x1
    target_height = target_y2 - target_y1
    target_width = target_x2 - target_x1
    
    mode_str = "Swapping" if swap else "Replacing"
    print(f"{mode_str} region:")
    print(f"  Source region: time [{source_start_frame}-{source_end_frame}] frames, spatial [{source_x1}:{source_x2}, {source_y1}:{source_y2}]")
    print(f"  Target region: time [{target_start_frame}-{target_end_frame}] frames, spatial [{target_x1}:{target_x2}, {target_y1}:{target_y2}]")
    print(f"  Source region size: {source_width}x{source_height}, Target region size: {target_width}x{target_height}")
    
    # Extract source and target regions
    if source_height > 0 and source_width > 0 and target_height > 0 and target_width > 0:
        source_patch = frames[source_start_frame:source_end_frame, 
                             source_y1:source_y2, 
                             source_x1:source_x2, :].copy()
        
        target_patch = frames[target_start_frame:target_end_frame,
                             target_y1:target_y2,
                             target_x1:target_x2, :].copy()
        
        # Resize source patch to match target dimensions if needed
        if source_height != target_height or source_width != target_width:
            print(f"  Resizing source patch from {source_width}x{source_height} to {target_width}x{target_height}")
            resized_patches = []
            for frame_idx in range(source_patch.shape[0]):
                resized_frame = cv2.resize(source_patch[frame_idx], (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                resized_patches.append(resized_frame)
            source_patch = np.array(resized_patches)
        
        # Resize target patch to match source dimensions if swapping
        if swap and (target_height != source_height or target_width != source_width):
            print(f"  Resizing target patch from {target_width}x{target_height} to {source_width}x{source_height}")
            resized_patches = []
            for frame_idx in range(target_patch.shape[0]):
                resized_frame = cv2.resize(target_patch[frame_idx], (source_width, source_height), interpolation=cv2.INTER_LINEAR)
                resized_patches.append(resized_frame)
            target_patch = np.array(resized_patches)
        
        # Replace target region with source patch
        num_target_frames = target_end_frame - target_start_frame
        num_source_frames = source_patch.shape[0]
        
        if num_target_frames > 0 and num_source_frames > 0:
            for i, target_frame_idx in enumerate(range(target_start_frame, target_end_frame)):
                # Cycle through source frames if target region is longer
                source_frame_idx = i % num_source_frames
                frames_modified[target_frame_idx, 
                               target_y1:target_y2, 
                               target_x1:target_x2, :] = source_patch[source_frame_idx]
            
            print(f"  Replaced {num_target_frames} frames in target region")
            
            # If swapping, also replace source region with target patch
            if swap:
                num_source_frames_to_replace = source_end_frame - source_start_frame
                num_target_frames_for_swap = target_patch.shape[0]
                
                for i, source_frame_idx in enumerate(range(source_start_frame, source_end_frame)):
                    # Cycle through target frames if source region is longer
                    target_frame_idx_for_swap = i % num_target_frames_for_swap
                    frames_modified[source_frame_idx,
                                   source_y1:source_y2,
                                   source_x1:source_x2, :] = target_patch[target_frame_idx_for_swap]
                
                print(f"  Replaced {num_source_frames_to_replace} frames in source region")
                print(f"  Swap completed!")
        else:
            print("Error: No source or target frames available")
    else:
        print("Error: Invalid source or target region dimensions")
    
    return frames_modified


if __name__ == '__main__':
    main()
