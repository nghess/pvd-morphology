import numpy as np
from scipy.spatial.distance import euclidean
from typing import List, Tuple, Dict

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

def match_segments(segments_list: List[List[List[Tuple[int, int, int]]]], dist_threshold: float = 15.0, manual_midpt: int = 15, auto_midpt: bool = False) -> List[Dict]:
    def segment_similarity(line1, line2):
        z1, x1, y1 = line1[-1]  # End of segment
        z1_, x1_, y1_ = line2[-1]
        if auto_midpt:
            z2, x2, y2 = line1[len(line1)//2]  # Sample a point halfway through the segment
            z2_, x2_, y2_ = line2[len(line2)//2]
        else:
            z2, x2, y2 = line1[manual_midpt-1]  # Sample a hardcoded point farther up the segment
            z2_, x2_, y2_ = line2[manual_midpt-1]
        
        dist_start = euclidean((z1, x1, y1), (z1_, x1_, y1_))
        dist_end = euclidean((z2, x2, y2), (z2_, x2_, y2_))
        
        return dist_start + dist_end

    def find_best_match(line, other_lines):
        best_match_index = None
        best_score = float('inf')
        
        for idx, other_line in enumerate(other_lines):
            score = segment_similarity(line, other_line)
            if score < best_score:
                best_score = score
                best_match_index = idx
        
        return best_match_index, best_score

    def assign_confidence(score, dist_threshold):
        if score < dist_threshold:
            return 'High'
        elif score < 2 * dist_threshold:
            return 'Medium'
        else:
            return 'Low'

    results = []
    
    for idx1, line in enumerate(segments_list[0]):
        result = {'line': line, 'index0': idx1}
        current_line = line
        
        for t in range(1, len(segments_list)):
            match_index, score = find_best_match(current_line, segments_list[t])
            confidence = assign_confidence(score, dist_threshold)
            
            if match_index is not None:
                match_line = segments_list[t][match_index]
            else:
                match_line = None
            
            result.update({
                f'match{t}': match_line,
                f'index{t}': match_index,
                f'confidence{t}': confidence,
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