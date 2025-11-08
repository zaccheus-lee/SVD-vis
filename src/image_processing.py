import numpy as np
from PIL import Image

def compress_image(image, k):
    """Compress image using SVD with k components."""
    img_array = np.array(image)
    compressed_channels = []
    
    for channel in range(3):
        channel_data = img_array[:, :, channel]
        U, S, Vt = np.linalg.svd(channel_data, full_matrices=False)
        compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        compressed_channels.append(compressed)
    
    compressed_image = np.stack(compressed_channels, axis=2)
    return compressed_image

def normalize_image(image):
    """Normalize image values to [0, 255] range."""
    image_min = image.min()
    image_max = image.max()
    normalized = (image - image_min) / (image_max - image_min) * 255
    return normalized.astype(np.uint8)
