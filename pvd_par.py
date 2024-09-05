import os
import time
import numpy as np
import pandas as pd
from skimage import io
import tifffile as tiff
from preprocess import preprocess_data
from skeletonize import skeletonize_data
from segment_matching import find_outer_segments, match_segments, get_matched_segments
from visualize import *
from pvd_metrics import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import KDTree


class PVD:
    def __init__(self, path, dataset, session, file):
        self.path = path
        self.dataset = dataset
        self.session = session
        self.file = file
        self.tiff_stack_path = f"{self.path}/{self.dataset}/{self.session}/{file}"
        self.raw_data = None
        self.processed_data = None
        self.mip_masks = None
        self.mip = None
        self.skeletonized_data = None  # Get rid of this at some point. Wasteful.
        self.skeleton_idx = None
        self.tips = None
        self.knots = None
        self.outer_segments = None
        self.matched_segments = None
        self.unmatched_segments = None
        self.matched_volumes = None
        self.labeled_data = []
        self.label_colors = []
        self.segment_dataframe = None

    def load_data(self):
        self.raw_data = io.imread(self.tiff_stack_path)

    def crop_data(self, size=2000):
        cropped_array = np.zeros((self.raw_data.shape[0], self.raw_data.shape[1], size, size))

        def crop_slice(image, new_size=size):
            assert new_size < image.shape[0], "New size must be smaller than original image"
            x_extra, y_extra = image.shape[0] - new_size, image.shape[1] - new_size
            x_1, y_1 = x_extra // 2, y_extra // 2
            return image[x_1:x_1+new_size, y_1:y_1+new_size]
        
        for ii in range(self.raw_data.shape[0]):
           cropped_array[ii,:,:,:] = np.array([crop_slice(self.raw_data[ii, z,:,:]) for z in range(self.raw_data.shape[1])])

        self.raw_data = cropped_array

    def preprocess_timepoint(self, timepoint):
        result = preprocess_data(self.raw_data[timepoint])
        return timepoint, result

    def preprocess(self):
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_data() first.")
        
        num_timepoints = self.raw_data.shape[0]
        self.processed_data = [None] * num_timepoints
        self.mip_masks = [None] * num_timepoints
        self.mip = [None] * num_timepoints
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.preprocess_timepoint, timepoint) 
                       for timepoint in range(num_timepoints)]
            
            for future in as_completed(futures):
                try:
                    timepoint, (preprocessed, mip_mask) = future.result()
                    self.processed_data[timepoint] = preprocessed
                    self.mip_masks[timepoint] = mip_mask
                    #print(f"Completed preprocessing timepoint {timepoint}")
                except Exception as exc:
                    print(f"Preprocessing generated an exception: {exc}")

        # Check if all timepoints were preprocessed
        missing = [i for i in range(num_timepoints) if self.processed_data[i] is None or self.mip_masks[i] is None]
        if missing:
            print(f"Warning: Timepoints {missing} were not preprocessed successfully")

        # Convert lists to numpy arrays
        self.processed_data = np.array(self.processed_data, dtype=object)
        #self.mip_masks = np.array(self.mip_masks, dtype=np.uint8)

        self.raw_data = "Raw data has been removed after preprocessing to save memory."  # Clear raw data

        # Create MIP for processed data
        for ii in range(num_timepoints):
            self.mip[ii] = np.max(self.processed_data[ii], axis=0)

    def skeletonize(self):
        if self.processed_data is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        
        results = skeletonize_data(self.processed_data)
        self.skeletonized_data, self.tips, self.knots, self.skeleton_idx = zip(*results)

    def find_outer_segments(self):
        if self.skeletonized_data is None or self.tips is None or self.knots is None:
            raise ValueError("Data not skeletonized. Call skeletonize() first.")
        
        self.outer_segments = []
        for t in range(len(self.skeletonized_data)):
            segments = find_outer_segments(self.skeletonized_data[t], self.tips[t], self.knots[t])
            self.outer_segments.append(segments)

    def match_segments(self):
        if self.outer_segments is None:
            raise ValueError("Outer segments not found. Call find_outer_segments() first.")
        
        matched_results = match_segments(self.outer_segments)
        self.matched_segments = get_matched_segments(self.outer_segments, matched_results)
        
        print(f"Matched {len(self.matched_segments[0])} segments across all timepoints.")

    def set_cells_to_zero(self, arr, segment_coords):
        def coord_generator():
            for sublist in segment_coords:
                for coord in sublist:
                    yield coord

        coords = np.array(list(coord_generator())).T

        if len(coords) > 0:
            arr[coords[0], coords[1], coords[2]] = 0
        
        return arr

    def get_unmatched_voxels(self):
        self.unmatched_segments = []
        for t in range(self.processed_data.shape[0]):
            self.unmatched_segments.append(self.set_cells_to_zero(self.skeletonized_data[t], self.matched_segments[t]))

        core_segments = []  # This list is populated with indices of all unmatched ('core') segments

        for ii in range(len(self.unmatched_segments)):
            core_segments.append(np.where(self.unmatched_segments[ii] == 1))

        for ii in range(len(self.unmatched_segments)):
            x = list(core_segments[ii][1])
            y = list(core_segments[ii][2])
            z = list(core_segments[ii][0])

            self.matched_segments[ii].append(list(zip(z,x,y)))

    def visualize_skeleton(self, output_path):
        if self.skeletonized_data is None or self.tips is None or self.knots is None:
            raise ValueError("Data not skeletonized. Call skeletonize() first.")
        
        for t in range(len(self.skeletonized_data)):
            visualize_skeleton(
                self.skeletonized_data[t],
                self.tips[t],
                self.knots[t],
                f'{output_path}/skeleton_t{t}.html',
                f'Skeleton: <b>{self.session}</b></br>Timepoint <b>{t}</b>'
            )
        
        print(f"Skeleton visualizations saved to {output_path}")

    def visualize_outer_segments(self, output_path):
        if self.outer_segments is None:
            raise ValueError("Outer segments not found. Call find_outer_segments() first.")
        
        for t in range(len(self.outer_segments)):
            visualize_segments(
                self.outer_segments[t],
                self.skeletonized_data[t],
                f'{output_path}/outer_segments_t{t}.html',
                f'Outer Segments: <b>{self.session}</b></br>Timepoint <b>{t}</b>'
            )
        
        print(f"Outer segment visualizations saved to {output_path}")

    def visualize_matched_segments(self, output_path):
        if self.matched_segments is None:
            raise ValueError("Matched segments not found. Call match_segments() first.")
        
        visualize_matched_segments(
            self.matched_segments,
            self.unmatched_segments,
            f'{output_path}/matched_segments.html',
            f'Matched Segments (4 timepoints): <b>{self.session}</b>'
        )

    def label_segmented_volume(self):
        if self.processed_data is None or self.matched_segments is None:
            raise ValueError("Processed data or matched segments not available. Make sure to run the pipeline first.")

        def find_indices(array):
            return np.argwhere(array > 0)

        def assign_segments_to_indices(indices, segments):
            flattened_segments = []
            coord_to_segment = {}
            
            for segment_index, segment in enumerate(segments):
                for coord in segment:
                    flattened_segments.append(coord)
                    coord_to_segment[tuple(coord)] = segment_index
            
            tree = KDTree(flattened_segments)
            _, nearest_indices = tree.query(indices)
            
            segment_assignments = [coord_to_segment[tuple(flattened_segments[i])] for i in nearest_indices]
            return segment_assignments

        def process_timepoint(timepoint):
            processed_data = self.processed_data[timepoint]
            data_idx = find_indices(processed_data)
            segment_assignments = assign_segments_to_indices(data_idx, self.matched_segments[timepoint])

            shape = processed_data.shape
            labeled_array = np.zeros(shape, dtype=np.uint16)
            labeled_array[data_idx[:, 0], data_idx[:, 1], data_idx[:, 2]] = np.array(segment_assignments) + 1  # +1 to reserve 0 for background

            return timepoint, labeled_array

        num_timepoints = len(self.processed_data)
        self.labeled_data = [None] * num_timepoints

        with ThreadPoolExecutor(max_workers=min(os.cpu_count(), num_timepoints)) as executor:
            futures = [executor.submit(process_timepoint, timepoint) 
                    for timepoint in range(num_timepoints)]
            
            for future in as_completed(futures):
                try:
                    timepoint, labeled_array = future.result()
                    self.labeled_data[timepoint] = labeled_array
                    #print(f"Timepoint {timepoint} volume labeled successfully.")
                except Exception as exc:
                    print(f"Labeling generated an exception for timepoint {timepoint}: {exc}")

        # Check if all timepoints were labeled
        missing = [i for i in range(num_timepoints) if self.labeled_data[i] is None]
        if missing:
            print(f"Warning: Timepoints {missing} were not labeled successfully")
        else:
            print("All timepoints labeled successfully")

        self.num_labels = max(np.max(arr) for arr in self.labeled_data)
        print(f"Number of unique labels: {self.num_labels}")

    def generate_dataframe(self):
        if self.labeled_data is None or self.matched_segments is None:
            raise ValueError("Labeled data or matched segments not available. Make sure to run label_segmented_volume() first.")

        # Get the number of timepoints and matched segments
        num_timepoints = len(self.labeled_data)
        num_segments = len(self.matched_segments[0])  # Assuming all timepoints have the same number of matched segments

        # Create a dictionary to store the cell counts and segment lengths
        data = {'Timepoint': list(range(num_timepoints))}
        
        for i in range(num_segments):
            data[f'Segment_{i+1}_Count'] = []
            data[f'Segment_{i+1}_Length'] = []

        # Count cells for each segment and get segment lengths at each timepoint
        for t in range(num_timepoints):
            labeled_volume = self.labeled_data[t]
            unique, counts = np.unique(labeled_volume, return_counts=True)
            count_dict = dict(zip(unique, counts))
            
            for i in range(num_segments):
                # Voxel count (volume)
                data[f'Segment_{i+1}_Count'].append(count_dict.get(i+1, 0))
                
                # Segment length
                segment_length = len(self.matched_segments[t][i])
                data[f'Segment_{i+1}_Length'].append(segment_length)

        # Create the DataFrame
        df = pd.DataFrame(data)

        # Set 'Timepoint' as the index
        df.set_index('Timepoint', inplace=True)

        self.segment_dataframe = df

        return df

    # Here we execute the full pipeline and log basic info and processing time
    def run_pipeline(self):
        def timer(start_time):
            return f"{time.time() - start_time:.2f} seconds"

        print("Starting pipeline.")
        total_start = time.time()

        start = time.time()
        self.load_data()
        print(f"Data loaded. Shape: {self.raw_data.shape}: {timer(start)}")

        start = time.time()
        self.crop_data()
        print(f"Data cropped. Shape: {self.raw_data.shape}: {timer(start)}")

        start = time.time()
        self.preprocess()
        print(f"Preprocessing complete: {timer(start)}")

        start = time.time()
        self.skeletonize()
        print(f"Data skeletonized: {timer(start)}")
        print(f"Number of tips per timepoint: {[len(tip) if tip is not None else 0 for tip in self.tips]}")
        print(f"Number of knots per timepoint: {[len(knot) if knot is not None else 0 for knot in self.knots]}")

        start = time.time()
        self.find_outer_segments()
        print(f"Outer segments found. Number of outer segments per timepoint: {[len(segments) for segments in self.outer_segments]}: {timer(start)}")

        start = time.time()
        self.match_segments()
        print(f"Segments matched. Number of matched segments per timepoint: {[len(segments) for segments in self.matched_segments]}: {timer(start)}")

        start = time.time()
        self.get_unmatched_voxels()
        print(f"Unmatched segments grouped: {timer(start)}")

        start = time.time()
        self.label_segmented_volume()
        print(f"Processed data labeled: {timer(start)}")

        start = time.time()
        self.generate_dataframe()
        print(f"Volume changes DataFrame generated: {timer(start)}")

        print(f"Pipeline complete. Total time: {timer(total_start)}")

    # Function to handle saving of all potential outputs from pipeline
    def save_results(self, output_path, save_numpy=True, save_tiff=False, save_plotly=True, save_labeled_tiff=False, save_dataframe=True):
        if self.processed_data is None or self.mip_masks is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        
        # Check is save path exists. If not, create required directories
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Save plot comparing MIP of each timepoint
        visualize_mips(self, display=False, save=True, output_dir=output_path)

        # Save MIP cosine similarity results:
        cosine_matrix, quality_score = analyze_cosine_similarity(self, output_path, save=True)
        print(f"\nQuality Score: {quality_score:.4f}")

        for timepoint in range(len(self.processed_data)):
            if self.processed_data[timepoint] is not None:
                if save_tiff:
                    tiff.imwrite(f'{output_path}/thresh_stack_{timepoint}.tif', self.processed_data[timepoint].astype(np.uint8)*255)
                if save_numpy:
                    np.save(f'{output_path}/pvd_binary_{timepoint}.npy', self.processed_data[timepoint].astype(np.uint8))
            if self.mip_masks[timepoint] is not None:
                if save_tiff:
                    tiff.imwrite(f'{output_path}/mip_mask_{timepoint}.tif', self.mip_masks[timepoint].astype(np.uint8)*255)
                if save_numpy:
                    np.save(f'{output_path}/mip_mask_{timepoint}.npy', self.mip_masks[timepoint])

        # Change this to pickle? Also perhaps just save the entire PVD object?
        if self.skeletonized_data is not None:
            for timepoint in range(len(self.skeletonized_data)):
                if self.skeletonized_data[timepoint] is not None:
                    if save_numpy:
                        np.save(f'{output_path}/pvd_skeleton_{timepoint}.npy', self.skeletonized_data[timepoint])
                if self.tips[timepoint] is not None:
                    if save_numpy:
                        np.save(f'{output_path}/pvd_tips_{timepoint}.npy', np.array(self.tips[timepoint]))
                if self.knots[timepoint] is not None:
                    if save_numpy:
                        np.save(f'{output_path}/pvd_knots_{timepoint}.npy', np.array(self.knots[timepoint]))

        if self.matched_segments is not None:
            for timepoint in range(len(self.matched_segments)):
                if save_numpy:
                    np.save(f'{output_path}/pvd_matched_segments_{timepoint}.npy', self.matched_segments[timepoint])

        # Save plotly visualizations
        if save_plotly:
            vis_path = f'{output_path}/visualizations'
            os.makedirs(vis_path, exist_ok=True)
            self.visualize_skeleton(vis_path)
            self.visualize_outer_segments(vis_path)
            self.visualize_matched_segments(vis_path)

        # Save labelled tiffs
        if save_labeled_tiff:
            if self.labeled_data is not None:
                for timepoint, labeled_array in enumerate(self.labeled_data):
                    colored_array = color_labeled_volume(labeled_array, self.num_labels)
                    tiff.imwrite(f'{vis_path}/labeled_data_t{timepoint}.tif', colored_array)
                print(f"Labeled data saved as color TIFF files in {output_path}")

        # Save volume changes DataFrame
        if save_dataframe and hasattr(self, 'segment_dataframe'):
            csv_path = f'{output_path}/segment_change.csv'
            self.segment_dataframe.to_csv(csv_path)
            print(f"Volume changes DataFrame saved to {csv_path}")