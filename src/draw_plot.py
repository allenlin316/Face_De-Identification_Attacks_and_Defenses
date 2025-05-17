import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


# +
# Draw the graph of comparison
def plot_epsilon_comparison(
    original_image,
    dp_pix_images: Dict[float, np.ndarray],
    dp_blur_images: Dict[float, np.ndarray],
    dp_processor, # DPImageObfuscator instance
    save_path: str = None
):
    """
    Plot SSIM and MSE comparison of DP-Pix and DP-Blur with different epsilon values.
    
    Args:
        original_image: Original image
        dp_pix_images: DP-Pix processed images with different epsilon values, format: {epsilon: image}
        dp_blur_images: DP-Blur processed images with different epsilon values, format: {epsilon: image}
        dp_processor: DPImageObfuscator instance, used for calculating metrics
        save_path: Path prefix to save the plots (optional)
    """
    # Get all epsilon values and sort them
    epsilon_values = sorted(set(list(dp_pix_images.keys()) + list(dp_blur_images.keys())))
    
    # Initialize result lists
    metrics = {
        'dp_pix': {'ssim': [], 'mse': []},
        'dp_blur': {'ssim': [], 'mse': []}
    }
    
    # Calculate metrics for each epsilon value
    for eps in epsilon_values:
        # Calculate metrics for DP-Pix
        if eps in dp_pix_images:
            mse, ssim = dp_processor.compute_metrics(original_image, dp_pix_images[eps])
            metrics['dp_pix']['mse'].append(mse)
            metrics['dp_pix']['ssim'].append(ssim)
            print(f"DP-Pix (ε={eps}): MSE={mse:.2f}, SSIM={ssim:.4f}")
        
        # Calculate metrics for DP-Blur
        if eps in dp_blur_images:
            mse, ssim = dp_processor.compute_metrics(original_image, dp_blur_images[eps])
            metrics['dp_blur']['mse'].append(mse)
            metrics['dp_blur']['ssim'].append(ssim)
            print(f"DP-Blur (ε={eps}): MSE={mse:.2f}, SSIM={ssim:.4f}")
    
    # Plot SSIM chart
    plt.figure(figsize=(10, 6))
    
    if dp_pix_images:
        plt.plot(epsilon_values, metrics['dp_pix']['ssim'], 'o-', color='blue', 
                linewidth=2, markersize=8, label='DP-Pix')
        
        # Add data labels
        for i, eps in enumerate(epsilon_values):
            if eps in dp_pix_images:
                ssim = metrics['dp_pix']['ssim'][i]
                plt.annotate(f'{ssim:.4f}', (eps, ssim), textcoords="offset points", 
                             xytext=(0, 10), ha='center', fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    if dp_blur_images:
        plt.plot(epsilon_values, metrics['dp_blur']['ssim'], 's-', color='green', 
                linewidth=2, markersize=8, label='DP-Blur')
        
        # Add data labels
        for i, eps in enumerate(epsilon_values):
            if eps in dp_blur_images:
                ssim = metrics['dp_blur']['ssim'][i]
                plt.annotate(f'{ssim:.4f}', (eps, ssim), textcoords="offset points", 
                             xytext=(0, -15), ha='center', fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.xlabel('Epsilon (ε)', fontsize=12)
    plt.ylabel('SSIM (higher is better)', fontsize=12)
    plt.title('SSIM vs Epsilon (ε)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}_ssim.png", dpi=300, bbox_inches='tight')
        print(f"SSIM plot saved to {save_path}_ssim.png")
    
    plt.show()
    
    # Plot MSE chart
    plt.figure(figsize=(10, 6))
    
    if dp_pix_images:
        plt.plot(epsilon_values, metrics['dp_pix']['mse'], 'o-', color='red', 
                linewidth=2, markersize=8, label='DP-Pix')
        
        # Add data labels
        for i, eps in enumerate(epsilon_values):
            if eps in dp_pix_images:
                mse = metrics['dp_pix']['mse'][i]
                plt.annotate(f'{mse:.2f}', (eps, mse), textcoords="offset points", 
                             xytext=(0, 10), ha='center', fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    if dp_blur_images:
        plt.plot(epsilon_values, metrics['dp_blur']['mse'], 's-', color='orange', 
                linewidth=2, markersize=8, label='DP-Blur')
        
        # Add data labels
        for i, eps in enumerate(epsilon_values):
            if eps in dp_blur_images:
                mse = metrics['dp_blur']['mse'][i]
                plt.annotate(f'{mse:.2f}', (eps, mse), textcoords="offset points", 
                             xytext=(0, -15), ha='center', fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.xlabel('Epsilon (ε)', fontsize=12)
    plt.ylabel('MSE (lower is better)', fontsize=12)
    plt.title('MSE vs Epsilon (ε)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_mse.png", dpi=300, bbox_inches='tight')
        print(f"MSE plot saved to {save_path}_mse.png")
    
    plt.show()
    
    # Print results table
    print("\nResults Summary:")
    print("-" * 60)
    print(f"{'Epsilon':<10} | {'DP-Pix SSIM':<12} | {'DP-Blur SSIM':<12} | {'DP-Pix MSE':<12} | {'DP-Blur MSE':<12}")
    print("-" * 60)
    
    for i, eps in enumerate(epsilon_values):
        pix_ssim = metrics['dp_pix']['ssim'][i] if eps in dp_pix_images else '-'
        pix_mse = metrics['dp_pix']['mse'][i] if eps in dp_pix_images else '-'
        blur_ssim = metrics['dp_blur']['ssim'][i] if eps in dp_blur_images else '-'
        blur_mse = metrics['dp_blur']['mse'][i] if eps in dp_blur_images else '-'
        
        # Format output
        pix_ssim_str = f"{pix_ssim:.4f}" if isinstance(pix_ssim, float) else pix_ssim
        pix_mse_str = f"{pix_mse:.2f}" if isinstance(pix_mse, float) else pix_mse
        blur_ssim_str = f"{blur_ssim:.4f}" if isinstance(blur_ssim, float) else blur_ssim
        blur_mse_str = f"{blur_mse:.2f}" if isinstance(blur_mse, float) else blur_mse
        
        print(f"{eps:<10.2f} | {pix_ssim_str:<12} | {blur_ssim_str:<12} | {pix_mse_str:<12} | {blur_mse_str:<12}")
    
    return metrics

def plot_methods_comparison(
    original_image,
    blur_image,
    pixel_image,
    dp_blur_image,
    dp_pixel_image,
    dp_processor, # DPImageObfuscator instance
    epsilon: float,
    save_path: str = None
):
    """
    Plot bar charts comparing MSE and SSIM of four methods.
    
    Args:
        original_image: Original image
        blur_image: Standard Gaussian blur processed image
        pixel_image: Standard pixelization processed image
        dp_blur_image: DP-Blur processed image
        dp_pixel_image: DP-Pix processed image
        dp_processor: DPImageObfuscator instance, used for calculating metrics
        epsilon: Epsilon value used
        save_path: Path to save the plot (optional)
    """
    # Calculate MSE and SSIM for all methods
    metrics = {
        'Method': ['Gaussian Blur', 'Pixelization', 'DP-Blur', 'DP-Pix'],
        'MSE': [],
        'SSIM': []
    }
    
    # Calculate metrics for standard Gaussian blur
    mse, ssim = dp_processor.compute_metrics(original_image, blur_image)
    metrics['MSE'].append(mse)
    metrics['SSIM'].append(ssim)
    print(f"Gaussian Blur: MSE={mse:.2f}, SSIM={ssim:.4f}")
    
    # Calculate metrics for standard pixelization
    mse, ssim = dp_processor.compute_metrics(original_image, pixel_image)
    metrics['MSE'].append(mse)
    metrics['SSIM'].append(ssim)
    print(f"Pixelization: MSE={mse:.2f}, SSIM={ssim:.4f}")
    
    # Calculate metrics for DP-Blur
    mse, ssim = dp_processor.compute_metrics(original_image, dp_blur_image)
    metrics['MSE'].append(mse)
    metrics['SSIM'].append(ssim)
    print(f"DP-Blur: MSE={mse:.2f}, SSIM={ssim:.4f}")
    
    # Calculate metrics for DP-Pix
    mse, ssim = dp_processor.compute_metrics(original_image, dp_pixel_image)
    metrics['MSE'].append(mse)
    metrics['SSIM'].append(ssim)
    print(f"DP-Pix: MSE={mse:.2f}, SSIM={ssim:.4f}")
    
    # Plot MSE and SSIM bar charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Set chart title
    fig.suptitle(f'Image Quality Metrics Comparison (ε = {epsilon})', fontsize=16)
    
    # Color settings
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    # MSE chart (lower is better)
    bars1 = ax1.bar(metrics['Method'], metrics['MSE'], color=colors)
    ax1.set_ylabel('Mean Square Error (Lower is Better)', fontsize=12)
    ax1.set_title('MSE Comparison', fontsize=14)
    ax1.set_xticklabels(metrics['Method'], rotation=45, ha='right')
    
    # Add value labels to the bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{metrics["MSE"][i]:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # SSIM chart (higher is better)
    bars2 = ax2.bar(metrics['Method'], metrics['SSIM'], color=colors)
    ax2.set_ylabel('Structural Similarity Index (Higher is Better)', fontsize=12)
    ax2.set_title('SSIM Comparison', fontsize=14)
    ax2.set_xticklabels(metrics['Method'], rotation=45, ha='right')
    ax2.set_ylim(0, 1.05)  # SSIM is typically in the range 0-1
    
    # Add value labels to the bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{metrics["SSIM"][i]:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the chart
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Methods comparison plot saved to {save_path}")
    
    plt.show()
    
    # Print results table
    print("\nMethods Comparison Summary:")
    print("-" * 50)
    print(f"{'Method':<15} | {'MSE':<10} | {'SSIM':<10}")
    print("-" * 50)
    
    for i, method in enumerate(metrics['Method']):
        print(f"{method:<15} | {metrics['MSE'][i]:<10.2f} | {metrics['SSIM'][i]:<10.4f}")
    
    return metrics


# -

if __name__ == "__main__":
    # Create processor instance for calculating metrics
    dp_processor = DPImageObfuscator()
    
    # Assuming you already have processed images
    # Example 1: Plot comparison across different epsilon values
    # Format: {epsilon: image}
    dp_pix_images = {}
    dp_blur_images = {}
    original_image = None
    
    # Replace this with your actual code to read or get images
    # ...
    
    # Plot epsilon comparison
    plot_epsilon_comparison(
        original_image,
        dp_pix_images,
        dp_blur_images,
        dp_processor,
        save_path="results/epsilon_comparison"
    )
    
    # Example 2: Plot comparison of four methods
    # Assuming you already have these four processed results
    blur_image = None
    pixel_image = None
    dp_blur_image = None
    dp_pixel_image = None
    
    # Replace this with your actual code to read or get images
    # ...
    
    # Plot methods comparison
    plot_methods_comparison(
        original_image,
        blur_image,
        pixel_image,
        dp_blur_image,
        dp_pixel_image,
        dp_processor,
        epsilon=0.5,
        save_path="results/methods_comparison.png"
    )
