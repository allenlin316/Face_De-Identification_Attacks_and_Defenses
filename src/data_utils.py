import cv2
import os
import matplotlib.pyplot as plt
# import self-created image obfuscator functions
from src.image_obfuscator import DPImageObfuscator


# Show the Image - grayscale
def plot_image(img, title=None):
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap="gray")
    if title:
        plt.title(title)
    plt.axis('off')
    plt.style.use('default')
    plt.show()


def plot_multiple_images(images, titles=None):
    """
    Plot multiple grayscale images side by side.
    
    Parameters:
        images (list of np.array): List of grayscale images.
        titles (list of str): Optional list of titles for each image.
    """
    n = len(images)
    plt.figure(figsize=(4 * n, 4))
    
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap="gray")
        if titles and i < len(titles):
            plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()



def read_image(image_path):
    _, file_extension = os.path.splitext(image_path)
    
    if file_extension.lower() == '.pgm':
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print(f"Input PGM image as grayscale, shape: {image.shape if image is not None else 'None'}")
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print(f"Input image as grayscale, shape: {image.shape if image is not None else 'None'}")
            
    return image


# Visualize comparison between original & processed images
def visualize_results(original, processed, method, save_path=None):
    """
    Visualize and optionally save the results.
    
    Args:
        original: Original image
        processed: Processed image
        method: Processing method name
        save_path: Optional path to save the visualization
    """

    dp_processor = DPImageObfuscator()
    
    # Compute metrics
    mse = dp_processor.compute_mse(original, processed)
    ssim = dp_processor.compute_ssim(original, processed)
    
    # Create figure
    plt.figure(figsize=(12, 5))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot processed image
    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title(f'{method}\nMSE: {mse:.2f}, SSIM: {ssim:.4f}')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path)
        
    plt.show()
