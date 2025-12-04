# -*- coding: utf-8 -*-
import numpy as np
import skimage.transform as transform

def command2action(command_ids, ratios, terminals):
    """
    Convert command IDs to actions that modify bounding box ratios.
    
    Args:
        command_ids: Array of command IDs for each image in batch
        ratios: Current bounding box ratios [xmin, ymin, xmax, ymax] for each image
        terminals: Terminal flags for each image (1 if done, 0 otherwise)
    
    Returns:
        ratios: Updated bounding box ratios
        terminals: Updated terminal flags
    """
    batch_size = len(command_ids)
    
    # Action mapping dictionary for cleaner code
    action_map = {
        0: [1, 1, -1, -1],   # Zoom in (move all edges inward)
        1: [0, 0, -1, -1],   # Shrink from right/bottom
        2: [1, 0, 0, -1],    # Adjust top-right
        3: [0, 1, -1, 0],    # Adjust bottom-left
        4: [1, 1, 0, 0],     # Move top-left inward
        5: [1, 0, 1, 0],     # Expand horizontally on left
        6: [-1, 0, -1, 0],   # Shrink horizontally
        7: [0, -1, 0, -1],   # Shrink vertically
        8: [0, 1, 0, 1],     # Expand vertically on bottom
        9: [0, 1, 0, -1],    # Adjust bottom edge
        10: [1, 0, -1, 0],   # Adjust horizontal edges
        11: [0, -1, 0, 1],   # Adjust vertical edges
        12: [-1, 0, 1, 0],   # Expand horizontally on right
        13: None             # Terminal action
    }
    
    for i in range(batch_size):
        if terminals[i] == 1:
            continue
            
        command_id = int(command_ids[i])
        
        if command_id == 13:
            terminals[i] = 1
        elif command_id in action_map:
            if action_map[command_id] is not None:
                ratios[i] += action_map[command_id]
        else:
            raise ValueError(f'Undefined command type: {command_id}')
        
        # Clip ratios to valid range [0, 20]
        ratios[i] = np.clip(ratios[i], 0, 20)
        
        # Check minimum box size (must be at least 4x4 units)
        if ratios[i, 2] - ratios[i, 0] <= 4 or ratios[i, 3] - ratios[i, 1] <= 4:
            terminals[i] = 1

    return ratios, terminals


def generate_bbox(input_np, ratios):
    """
    Generate absolute bounding box coordinates from normalized ratios.
    
    Args:
        input_np: List of input images (numpy arrays)
        ratios: Normalized bounding box ratios (0-20 scale)
    
    Returns:
        bbox: List of bounding boxes [(xmin, ymin, xmax, ymax), ...]
    """
    assert len(input_np) == len(ratios), \
        f"Input length ({len(input_np)}) must match ratios length ({len(ratios)})"

    bbox = []
    for im, ratio in zip(input_np, ratios):
        height, width = im.shape[:2]
        
        # Convert normalized ratios (0-20) to pixel coordinates
        xmin = int(float(ratio[0]) / 20 * width)
        ymin = int(float(ratio[1]) / 20 * height)
        xmax = int(float(ratio[2]) / 20 * width)
        ymax = int(float(ratio[3]) / 20 * height)
        
        # Ensure valid bounds
        xmin = max(0, min(xmin, width))
        ymin = max(0, min(ymin, height))
        xmax = max(xmin + 1, min(xmax, width))
        ymax = max(ymin + 1, min(ymax, height))
        
        bbox.append((xmin, ymin, xmax, ymax))
        
    return bbox


def crop_input(input_np, bbox):
    """
    Crop images according to bounding boxes and resize to fixed size.
    
    Args:
        input_np: List of input images (numpy arrays)
        bbox: List of bounding boxes [(xmin, ymin, xmax, ymax), ...]
    
    Returns:
        result: Array of cropped and resized images (227x227)
    """
    assert len(input_np) == len(bbox), \
        f"Input length ({len(input_np)}) must match bbox length ({len(bbox)})"
    
    result = []
    for im, (xmin, ymin, xmax, ymax) in zip(input_np, bbox):
        # Crop the image
        cropped = im[ymin:ymax, xmin:xmax]
        
        # Handle edge case of empty crop
        if cropped.size == 0:
            # Use entire image if crop is invalid
            cropped = im
        
        # Resize to 227x227 (AlexNet input size)
        resized = transform.resize(cropped, (227, 227), 
                                   mode='constant', 
                                   anti_aliasing=True,
                                   preserve_range=False)
        result.append(resized)
    
    return np.asarray(result, dtype=np.float32)