import numpy as np
from scipy.spatial.distance import euclidean
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr

def cosine_sim(list1, list2):
    list1 = np.array(list1).flatten()
    list2 = np.array(list2).flatten()
    return cosine_similarity([list1], [list2])[0][0]


def find_filter_plane(skeleton):  # Locate the window with highest density of voxels and take center slice as filter plane
    plane_list = []
    for ii in range(1, skeleton.shape[0]-1):  # 3-slice window
        plane_list.append(np.mean(skeleton[ii-1:ii+1,:,:]))
    
    primary_plane = plane_list.index(max(plane_list))
    return primary_plane

def filter_outer_segments(outer_segments, filter_value, tolerance=10):  # Check if segment end is within tolerance of filter plane
    filtered_segments = [segment for segment in outer_segments if filter_value - tolerance <= segment[-1][0] <= filter_value + tolerance]
    return filtered_segments

def find_outer_segments(skeleton, tips, knots, filter_plane=True, min_length=True, target_length=15):
    def check_neighbors(array, x, y, z, mode="count", prev=None):
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

    def traverse_segment_breadth_first(array, start_indices, stop_indices):
        segments = []
        
        def bfs(start):  # Breadth-first search
            queue = [(start, [start])]
            visited = set([start])
            
            while queue:
                current, path = queue.pop(0)
                x, y, z = current
                
                if current in stop_indices:
                    segments.append(path)
                    return
                
                for neighbor in check_neighbors(array, x, y, z, mode="retrieve"):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))


        for start in start_indices:
            bfs(start)

        if filter_plane:
            fp_idx = find_filter_plane(skeleton)
            return filter_outer_segments(segments, fp_idx)
        else:
            return segments

    outer_segments = traverse_segment_breadth_first(skeleton, tips, knots)
    
    if min_length:
        return [segment for segment in outer_segments if len(segment) >= target_length]
    else:
        return outer_segments

def match_segments(segments_list: List[List[List[Tuple[int, int, int]]]], 
                            dist_threshold: float = 15.0, 
                            shape_weight: float = 0.4, 
                            location_weight: float = 0.6) -> List[Dict]:
    def segment_similarity(line1, line2):
        # Location similarity
        start_dist = euclidean(line1[0], line2[0])
        end_dist = euclidean(line1[-1], line2[-1])
        location_sim = 1 / (1 + (start_dist + end_dist) / 2)
        
        # Shape similarity
        # Resample segments to have the same number of points
        n_points = 50
        line1_resampled = np.array([line1[int(i * (len(line1) - 1) / (n_points - 1))] for i in range(n_points)])
        line2_resampled = np.array([line2[int(i * (len(line2) - 1) / (n_points - 1))] for i in range(n_points)])
        
        # Calculate shape similarity using Pearson correlation of coordinates
        corr_x = pearsonr(line1_resampled[:, 0], line2_resampled[:, 0])[0]
        corr_y = pearsonr(line1_resampled[:, 1], line2_resampled[:, 1])[0]
        corr_z = pearsonr(line1_resampled[:, 2], line2_resampled[:, 2])[0]
        shape_sim = (corr_x + corr_y + corr_z) / 3
        
        # Combine location and shape similarity
        return location_weight * location_sim + shape_weight * shape_sim

    def find_best_match(line, other_lines):
        best_match_index = None
        best_score = float('-inf')  # Changed to -inf as we're maximizing similarity
        
        for idx, other_line in enumerate(other_lines):
            score = segment_similarity(line, other_line)
            if score > best_score:
                best_score = score
                best_match_index = idx
        
        return best_match_index, best_score

    def assign_confidence(score):
        if score > 0.9:
            return 'High'
        elif score > 0.66:
            return 'Medium'
        else:
            return 'Low'

    results = []
    
    for idx1, line in enumerate(segments_list[0]):
        result = {'line': line, 'index0': idx1}
        current_line = line
        
        for t in range(1, len(segments_list)):
            match_index, score = find_best_match(current_line, segments_list[t])
            confidence = assign_confidence(score)
            
            if match_index is not None:
                match_line = segments_list[t][match_index]
            else:
                match_line = None
            
            result.update({
                f'match{t}': match_line,
                f'index{t}': match_index,
                f'confidence{t}': confidence,
                f'score{t}': score
            })
            
            if match_line is not None:
                current_line = match_line
            else:
                break
        
        results.append(result)
    
    return [r for r in results if all(r[f'confidence{t}'] != 'Low' for t in range(1, len(segments_list)))]

def get_matched_segments(segments_list: List[List[List[Tuple[int, int, int]]]], matched_results: List[Dict]) -> List[List[List[Tuple[int, int, int]]]]:
    matched_segments = [[] for _ in range(len(segments_list))]
    
    for result in matched_results:
        for t in range(len(segments_list)):
            if result[f'index{t}'] is not None:
                matched_segments[t].append(segments_list[t][result[f'index{t}']])
    
    return matched_segments