import os
import numpy as np
from skimage import io
import tifffile as tiff
from preprocess import preprocess_stack
from skeletonize import skeletonize_data
from concurrent.futures import ThreadPoolExecutor, as_completed

class PVD:
    def __init__(self, tiff_stack_path):
        self.tiff_stack_path = tiff_stack_path
        self.raw_data = None
        self.preprocessed_data = None
        self.mip_masks = None
        self.mip = None
        self.skeletonized_data = None
        self.tips = None
        self.knots = None

    def load_data(self):
        self.raw_data = io.imread(self.tiff_stack_path)
        print(f"Loaded raw data with shape: {self.raw_data.shape}")

    def preprocess_timepoint(self, timepoint):
        result = preprocess_stack(self.raw_data[timepoint])
        return timepoint, result

    def preprocess(self):
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_data() first.")
        
        num_timepoints = self.raw_data.shape[0]
        self.preprocessed_data = [None] * num_timepoints
        self.mip_masks = [None] * num_timepoints
        self.mip = [None] * num_timepoints
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.preprocess_timepoint, timepoint) 
                       for timepoint in range(num_timepoints)]
            
            for future in as_completed(futures):
                try:
                    timepoint, (preprocessed, mip_mask) = future.result()
                    self.preprocessed_data[timepoint] = preprocessed
                    self.mip_masks[timepoint] = mip_mask
                    #print(f"Completed preprocessing timepoint {timepoint}")
                except Exception as exc:
                    print(f"Preprocessing generated an exception: {exc}")

        # Check if all timepoints were preprocessed
        missing = [i for i in range(num_timepoints) if self.preprocessed_data[i] is None or self.mip_masks[i] is None]
        if missing:
            print(f"Warning: Timepoints {missing} were not preprocessed successfully")

        # Convert lists to numpy arrays
        self.preprocessed_data = np.array(self.preprocessed_data, dtype=object)
        self.mip_masks = np.array(self.mip_masks, dtype=object)

        # Create MIP for processed data
        for ii in range(num_timepoints):
            self.mip[ii] = np.max(self.preprocessed_data[ii], axis=0)

    def skeletonize(self):
        if self.preprocessed_data is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        
        results = skeletonize_data(self.preprocessed_data)
        self.skeletonized_data, self.tips, self.knots = zip(*results)
        
        print(f"Skeletonized data shape: {[skeleton.shape if skeleton is not None else None for skeleton in self.skeletonized_data]}")
        print(f"Number of tips per timepoint: {[len(tip) if tip is not None else 0 for tip in self.tips]}")
        print(f"Number of knots per timepoint: {[len(knot) if knot is not None else 0 for knot in self.knots]}")

    def save_results(self, output_path, save_npy=True, save_tiff=False):
        if self.preprocessed_data is None or self.mip_masks is None:
            raise ValueError("Data not preprocessed. Call preprocess() first.")
        
        # Check is save path exists. If not, create required directories
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for timepoint in range(len(self.preprocessed_data)):
            if self.preprocessed_data[timepoint] is not None:
                if save_tiff:
                    tiff.imwrite(f'{output_path}/thresh_stack_{timepoint}.tif', self.preprocessed_data[timepoint].astype(np.uint8)*255)
                if save_npy:
                    np.save(f'{output_path}/pvd_binary_{timepoint}.npy', self.preprocessed_data[timepoint])
            if self.mip_masks[timepoint] is not None:
                if save_tiff:
                    tiff.imwrite(f'{output_path}/mip_mask_{timepoint}.tif', self.mip_masks[timepoint].astype(np.uint8)*255)
                if save_npy:
                    np.save(f'{output_path}/mip_mask_{timepoint}.npy', self.mip_masks[timepoint])

        # Change this to pickle. Also perhaps just save the entire PVD object?
        if self.skeletonized_data is not None:
            for timepoint in range(len(self.skeletonized_data)):
                if self.skeletonized_data[timepoint] is not None:
                    if save_npy:
                        np.save(f'{output_path}/pvd_skeleton_{timepoint}.npy', self.skeletonized_data[timepoint])
                if self.tips[timepoint] is not None:
                    if save_npy:
                        np.save(f'{output_path}/pvd_tips_{timepoint}.npy', np.array(self.tips[timepoint]))
                if self.knots[timepoint] is not None:
                    if save_npy:
                        np.save(f'{output_path}/pvd_knots_{timepoint}.npy', np.array(self.knots[timepoint]))

    def run_pipeline(self):
        print("Starting pipeline")
        self.load_data()
        self.preprocess()
        self.skeletonize()
        print("Pipeline completed")