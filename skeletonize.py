import numpy as np
import operator
from skimage import filters, morphology
from scipy import ndimage
from scipy.ndimage import label
from concurrent.futures import ThreadPoolExecutor, as_completed

def find_indices(array, condition=operator.eq, value=1):
    indices = np.argwhere(condition(array, value))
    return [tuple(idx) for idx in indices]

def check_neighbors(array, x, y, z, mode="count", prev=None):
    assert mode in ["count", "retrieve"], "Mode must be either 'count' or 'retrieve'"
    count = 0
    neighbors = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if i == 0 and j == 0 and k == 0:
                    continue
                nx, ny, nz = x + i, y + j, z + k
                if 0 <= nx < array.shape[0] and 0 <= ny < array.shape[1] and 0 <= nz < array.shape[2]:
                    if array[nx, ny, nz] != 0 and (nx, ny, nz) != prev:
                        count += 1
                        neighbors.append((nx, ny, nz))
    return count if mode == "count" else neighbors

def find_nodes(array, indices, condition=operator.eq, neighbor_criterion=1):
    cells_matching_criterion = []
    for z, x, y in indices:
        if array[z, x, y] != 0 and condition(check_neighbors(array, z, x, y, mode="count"), neighbor_criterion):
            cells_matching_criterion.append((z, x, y))
    return cells_matching_criterion

def check_tolerance(coord1, coord2, tolerance):
    return all(abs(a - b) <= tolerance for a, b in zip(coord1, coord2))

def filter_coordinates(list_a, list_b, tolerance=5):
    return [coord_a for coord_a in list_a if not any(check_tolerance(coord_a, coord_b, tolerance) for coord_b in list_b)]

def remove_close_coordinates(coords, tolerance=5):
    filtered_coords = []
    for coord in coords:
        if all(not check_tolerance(coord, existing, tolerance) for existing in filtered_coords):
            filtered_coords.append(coord)
    return filtered_coords

def remove_floating_segments(array):
    structuring_element = np.ones((3, 3, 3), dtype=int)
    labeled_array, num_features = label(array, structure=structuring_element)
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0
    largest_label = sizes.argmax()
    return (labeled_array == largest_label).astype(int)

def skeletonize_stack(binary_stack):
    # Ensure the input is binary
    binary_stack = binary_stack.astype(bool)
    
    # Dilate
    dilated = morphology.binary_dilation(binary_stack, morphology.ball(radius=3))
    
    # Apply Gaussian smoothing
    sigma = 3
    smoothed = filters.gaussian(dilated, sigma=sigma)
    
    # Threshold back to binary
    thresh = filters.threshold_otsu(smoothed)
    binary = smoothed > thresh
    
    # Skeletonize
    skeleton = morphology.skeletonize(binary, method='lee')
    
    # Remove floating segments
    skeleton_main = remove_floating_segments(skeleton)
    
    return skeleton_main

def find_tips_and_knots(skeleton):
    skeleton_idx = find_indices(skeleton)
    tips = find_nodes(skeleton, skeleton_idx, condition=operator.eq, neighbor_criterion=1)
    knots = find_nodes(skeleton, skeleton_idx, condition=operator.ge, neighbor_criterion=3)
    knots = remove_close_coordinates(knots, tolerance=1)
    tips = filter_coordinates(tips, knots, tolerance=5)
    
    return tips, knots

def process_timepoint(timepoint, binary_data):
    try:
        print(f"Skeletonizing timepoint {timepoint}")
        skeleton = skeletonize_stack(binary_data)
        tips, knots = find_tips_and_knots(skeleton)
        return timepoint, skeleton, tips, knots
    except Exception as e:
        print(f"Error skeletonizing timepoint {timepoint}: {str(e)}")
        return timepoint, None, None, None

def skeletonize_data(binary_data):
    num_timepoints = len(binary_data)
    results = [None] * num_timepoints
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_timepoint, i, data) for i, data in enumerate(binary_data)]
        
        for future in as_completed(futures):
            try:
                timepoint, skeleton, tips, knots = future.result()
                results[timepoint] = (skeleton, tips, knots)
                if skeleton is not None:
                    print(f"Completed skeletonizing timepoint {timepoint}")
                else:
                    print(f"Failed to skeletonize timepoint {timepoint}")
            except Exception as exc:
                print(f'Future for timepoint skeletonization generated an exception: {exc}')
    
    # Check if all timepoints were processed
    failed_timepoints = [i for i, r in enumerate(results) if r[0] is None]
    if failed_timepoints:
        print(f"Warning: Timepoints {failed_timepoints} were not skeletonized successfully")
    
    return results