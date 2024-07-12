import numpy as np
import cv2
from skimage import morphology
from scipy.ndimage import binary_erosion, binary_dilation, label

def normalize_img(image, dtype=np.uint8, ceiling=255):
    image = image.astype(float)
    image -= np.amin(image)
    image /= np.amax(image)
    image *= ceiling
    return image.astype(dtype)

def set_black(image, dtype=np.uint8):
    window = image[:250,:250]
    black_pt = np.mean(window)
    thresholded_image = np.maximum(image, black_pt)
    scale_factor = 255 / (255 - black_pt)
    adjusted_image = (thresholded_image - black_pt) * scale_factor
    return np.clip(adjusted_image, 0, 255).astype(np.uint8)

def sigmoid_adjustment(image, std_mult=1.5, alpha=0.5):
    mean_val = np.mean(image)
    std_dev = np.std(image)
    midpoint = mean_val + std_dev * std_mult
    adjusted_image = 1 / (1 + np.exp(-alpha * (image - midpoint)))
    return np.uint8(255 * (adjusted_image - adjusted_image.min()) / (adjusted_image.max() - adjusted_image.min()))  # Do we really want to go backto 255?

def mask_slice(image, mask):
    return image * mask

def threshold_slice(image, threshold=0, cleaned=False, min_size=4):  # Is this being used?
    canvas = np.ones_like(image)
    image = np.where(image > threshold, canvas, 0)
    if cleaned:
        binary_img = image > 0
        image = morphology.remove_small_objects(binary_img, min_size, connectivity=2)
    return image

def process_stack(input_stack, process_func, *args, **kwargs):  # Different from notebook, seems okay though
    return np.array([process_func(input_stack[z,:,:], *args, **kwargs) for z in range(input_stack.shape[0])])

def create_mip_mask(input_stack, dilation_radius=3, erosion_radius=2):
    mip_image = np.max(input_stack, axis=0)
    mip_image_norm = normalize_img(mip_image, ceiling=1)
    mip_mask = np.where(normalize_img(mip_image) >= 4, 1, 0)
    mip_mask = morphology.binary_dilation(mip_mask, morphology.disk(radius=dilation_radius))
    mip_mask = morphology.binary_erosion(mip_mask, morphology.disk(radius=erosion_radius))
    return morphology.remove_small_objects(mip_mask > 0, min_size=48, connectivity=4)

def remove_floating_regions(thresh_stack, connectivity=26):
    structuring_element = np.ones((3, 3, 3), dtype=bool)
    #opened_array = binary_dilation(binary_erosion(thresh_stack, structure=structuring_element), structure=structuring_element)
    return morphology.remove_small_objects(thresh_stack > 0, min_size=48000, connectivity=connectivity)

def preprocess_stack(input_stack):
    norm_stack = process_stack(input_stack, normalize_img)
    black_stack = process_stack(norm_stack, set_black)
    processed_stack = process_stack(black_stack, sigmoid_adjustment, std_mult=3, alpha=1)
    mip_mask = create_mip_mask(input_stack)
    masked_stack = process_stack(processed_stack, mask_slice, mask=mip_mask)
    thresh_stack = process_stack(masked_stack, threshold_slice, threshold=64, min_size=4)
    final_stack = remove_floating_regions(thresh_stack)
    return final_stack.astype(np.uint8), mip_mask
