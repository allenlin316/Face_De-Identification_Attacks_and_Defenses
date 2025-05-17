import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional

# import self-created image obfuscator functions
from src.image_obfuscator import DPImageObfuscator, apply_pixelization, apply_gaussian_blur, apply_dp_pixelization, apply_dp_blur
from src.data_utils import plot_image, read_image
from src.draw_plot import plot_epsilon_comparison, plot_methods_comparison


def process_whole_image_with_dp(image_path='my_img.pgm', method='dp_pix', 
                          epsilon=0.5, m=16, block_size=10, 
                          kernel_size=(10, 10), sigma=0):
    """
    
    parameters:
        image_path: image path
        method: different image obfuscation ('dp_pix', 'dp_blur', 'pixelize', 'blur')
        epsilon: differential privacy epsilon
        m: neighbor image pixel difference
        block_size: pixelized block size
        kernel_size: gaussian kernel size
        sigma: sigma 
    
    return:
        processed image
    """
    # input image
    original_image = read_image(image_path)
    
    # 检查图像是否正确加载
    if original_image is None:
        print(f'Error: Unable to read image {image_path}')
        return None
    
    # 获取图像尺寸
    height, width = original_image.shape
    
    # 显示原始灰度图像
    #print("原始灰度图像：")
    # plot_image(original_image, "原始图像")
    
    if method == 'dp_pix':
        result_image = apply_dp_pixelization(original_image, epsilon, m, block_size)
        effect_name = "DP-Pix"
    elif method == 'dp_blur':
        result_image = apply_dp_blur(original_image, epsilon, m, kernel_size, sigma)
        effect_name = "DP-Blur"
    elif method == 'pixelize':
        result_image = apply_pixelization(original_image, block_size)
        effect_name = "Pixelize"
    elif method == 'blur':
        result_image = apply_gaussian_blur(original_image, kernel_size, sigma)
        effect_name = "Blur"
    else:
        print(f'Error: Unknown method {method}')
        return None
    
    # 创建处理器实例来计算指标
    dp_processor = DPImageObfuscator(epsilon=epsilon, m=m)
    
    # 计算MSE和SSIM
    mse, ssim_value = dp_processor.compute_metrics(original_image, result_image)
    print(f'MSE: {mse:.2f}, SSIM: {ssim_value:.4f}')
    
    # 显示处理后的图像
    print(f'Apply {effect_name} grayscale image:')
    # plot_image(result_image, f"After {effect_name}")
    
    return result_image


# +
# save all methods(pixelized, blur, DP-Pix, DP-Blur) images 
# hyperparameters
kernel_size = 3
block_size = 4
epsilon = 0.5
# Paths
source_root = os.path.join("..", "att_faces")           
destination_root_gaussian = os.path.join("..", "gaussian_faces", f"kernel_size_{kernel_size}") 
destination_root_pixelized = os.path.join("..", "pixelized_faces", f"block_size_{block_size}")
destination_root_dp_pixelized = os.path.join("..", "dp_pixelized_faces", f"block_size_{block_size}", f"epsilon_{epsilon}")
destination_root_dp_gaussian = os.path.join("..", "dp_gaussian_faces", f"kernel_size_{kernel_size}", f"epsilon_{epsilon}")

# Make sure destination root exists
os.makedirs(destination_root_gaussian, exist_ok=True)
os.makedirs(destination_root_pixelized, exist_ok=True)
os.makedirs(destination_root_dp_gaussian, exist_ok=True)
os.makedirs(destination_root_dp_pixelized, exist_ok=True)

for root, dirs, files in os.walk(source_root):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pgm')):  # image formats
            source_path = os.path.join(root, file)

            # Compute the relative path and destination path
            relative_path = os.path.relpath(source_path, source_root)
            destination_path_gaussian = os.path.join(destination_root_gaussian, relative_path)
            destination_path_pixelized = os.path.join(destination_root_pixelized, relative_path)
            destination_path_dp_gaussian = os.path.join(destination_root_dp_gaussian, relative_path)
            destination_path_dp_pixelized = os.path.join(destination_root_dp_pixelized, relative_path)
        
            # Ensure destination subfolder exists
            os.makedirs(os.path.dirname(destination_path_gaussian), exist_ok=True)
            os.makedirs(os.path.dirname(destination_path_pixelized), exist_ok=True)
            os.makedirs(os.path.dirname(destination_path_dp_gaussian), exist_ok=True)
            os.makedirs(os.path.dirname(destination_path_dp_pixelized), exist_ok=True)
            # Process the image (example: convert to grayscale)
            # Apply Gaussian
            print(f"Applied Gaussian on {relative_path}")
            blur_image = process_whole_image_with_dp(
                image_path=source_path,
                method='blur',
                kernel_size=(kernel_size, kernel_size)
            )
            # Apply Pixelize
            print(f"Applied Pixelized on {relative_path}")
            pixel_image = process_whole_image_with_dp(
                image_path=source_path,
                method='pixelize',
                block_size=block_size
            )
            print(f'Applied DP-Pix on {relative_path}')
            dp_pixel_image = process_whole_image_with_dp(
                image_path=source_path,
                method='dp_pix',
                epsilon=epsilon,
                m=16,
                block_size=block_size
            )
            print(f'Applied DP-Blur on {relative_path}')
            dp_blur_image = process_whole_image_with_dp(
                image_path=source_path,
                method='dp_blur',
                epsilon=epsilon,
                m=16,
                kernel_size=(kernel_size, kernel_size)
            )
            # saving all images
            #cv2.imwrite(destination_path_gaussian, blur_image)
            #cv2.imwrite(destination_path_pixelized, pixel_image)
            #cv2.imwrite(destination_path_dp_pixelized, dp_pixel_image)
            #cv2.imwrite(destination_path_dp_gaussian, dp_blur_image)
            

# +
# To visualize SSIM/MSE comparison of result images 

# Create processor instance for calculating metrics
dp_processor = DPImageObfuscator()

# Read original image
source_path = os.path.join("..", "att_faces", "s1", "1.pgm")
original_image = read_image(source_path)

# Process images for each epsilon value and store results
kernel_size = 3
block_size = 4
epsilon_values = [0.1, 0.5, 1.0]
dp_pix_images = {}
dp_blur_images = {}

for epsilon in epsilon_values:
    # Process DP-Pix
    print(f'Processing DP-Pix with ε={epsilon}')
    dp_pixel_image = process_whole_image_with_dp(
        image_path=source_path,
        method='dp_pix',
        epsilon=epsilon,
        m=16,
        block_size=block_size
    )
    dp_pix_images[epsilon] = dp_pixel_image
    
    # Process DP-Blur
    print(f'Processing DP-Blur with ε={epsilon}')
    dp_blur_image = process_whole_image_with_dp(
        image_path=source_path,
        method='dp_blur',
        epsilon=epsilon,
        m=16,
        kernel_size=(kernel_size, kernel_size)
    )
    dp_blur_images[epsilon] = dp_blur_image

# Plot epsilon comparison graphs
plot_epsilon_comparison(
    original_image,
    dp_pix_images,
    dp_blur_images,
    dp_processor,
    save_path="results/epsilon_comparison"
)

# Process standard methods
print("Processing standard Gaussian blur")
blur_image = process_whole_image_with_dp(
    image_path=source_path,
    method='blur',
    kernel_size=(kernel_size, kernel_size)
)

print("Processing standard pixelization")
pixel_image = process_whole_image_with_dp(
    image_path=source_path,
    method='pixelize',
    block_size=block_size
)

# Choose a specific epsilon value for method comparison
epsilon = 0.5
dp_pixel_image = dp_pix_images[epsilon]
dp_blur_image = dp_blur_images[epsilon]

# Plot method comparison bar charts
plot_methods_comparison(
    original_image,
    blur_image,
    pixel_image,
    dp_blur_image,
    dp_pixel_image,
    dp_processor,
    epsilon=epsilon,
    save_path=f"results/methods_comparison_eps{epsilon}.png"
)
# -


