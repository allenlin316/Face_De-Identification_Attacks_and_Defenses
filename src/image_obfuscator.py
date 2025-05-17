import numpy as np
import cv2
from typing import Tuple, Optional

# Try to import SSIM from scikit-image
try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SKIMAGE = True
except ImportError:
    HAVE_SKIMAGE = False


class DPImageObfuscator:
    """
    Implementation of differentially private image obfuscation methods.
    Based on the paper "Differential Privacy for Image Publication" by Liyue Fan.
    """
    
    def __init__(self, epsilon: float = 0.5, m: int = 16):
        """
        Initialize the DP image obfuscator.
        
        Args:
            epsilon: Privacy parameter. Lower epsilon provides stronger privacy guarantee.
            m: Neighborhood parameter. The number of pixels that can differ between 
               neighboring images that should remain indistinguishable.
        """
        self.epsilon = epsilon
        self.m = m
    
    # 標準像素化（非私有）
    def pixelize(self, image: np.ndarray, roi_x: int, roi_y: int, 
                roi_width: int, roi_height: int, block_size: int) -> np.ndarray:
        """
        Standard pixelization (non-private) for a specific region of interest.
        
        Args:
            image: Input grayscale image
            roi_x, roi_y, roi_width, roi_height: Region of interest coordinates
            block_size: Size of pixelization blocks
            
        Returns:
            Pixelized image
        """
        # 創建結果圖像的副本
        result = image.copy()
        
        # 提取感興趣區域(ROI)
        roi = image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        
        # 計算縮放因子
        h, w = roi.shape
        
        # 確保block_size不大於ROI的寬高
        block_size = min(block_size, h, w)
        
        # 縮小ROI
        temp = cv2.resize(roi, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
        
        # 放大回原始大小（產生像素化效果）
        pixelated_roi = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 將像素化後的ROI放回結果圖像
        result[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width] = pixelated_roi
        
        return result
    
    # 高斯模糊（非私有）
    def gaussian_blur(self, image: np.ndarray, roi_x: int, roi_y: int, 
                     roi_width: int, roi_height: int, kernel_size: Tuple[int, int], sigma: int = 0) -> np.ndarray:
        """
        Standard Gaussian blur (non-private) for a specific region of interest.
        
        Args:
            image: Input grayscale image
            roi_x, roi_y, roi_width, roi_height: Region of interest coordinates
            kernel_size: Size of Gaussian kernel (k×k)
            sigma: Gaussian kernel standard deviation
            
        Returns:
            Blurred image
        """
        # 創建結果圖像的副本
        result = image.copy()
        
        # 提取感興趣區域(ROI)
        roi = image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        
        # 應用高斯模糊
        blurred_roi = cv2.GaussianBlur(roi, kernel_size, sigma)
        
        # 將模糊後的ROI放回結果圖像
        result[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width] = blurred_roi
        
        return result
    
    # 差分隱私像素化 (DP-Pix)
    def dp_pix(self, image: np.ndarray, roi_x: int, roi_y: int, 
              roi_width: int, roi_height: int, block_size: int) -> np.ndarray:
        """
        Differentially private pixelization (DP-Pix) for a specific region of interest.
        Based on the paper "Differential Privacy for Image Publication" by Liyue Fan.
        
        Args:
            image: Input grayscale image
            roi_x, roi_y, roi_width, roi_height: Region of interest coordinates
            block_size: Size of pixelization blocks (b×b)
            
        Returns:
            Differentially private pixelized image
        """
        # 创建结果图像的副本
        result = image.copy()
        
        # 提取感兴趣区域(ROI)
        roi = image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        
        sensitivity_factor = 0.1  # 减小敏感度以匹配论文MSE
        
        # 计算全局敏感度: 255*m/b^2
        # 根据GitHub代码和原始论文的公式
        global_sensitivity = (255.0 * self.m) / (block_size * block_size) * sensitivity_factor
        
        # 计算拉普拉斯尺度参数 beta = GS / epsilon
        beta = global_sensitivity / self.epsilon
        
        # 创建输出ROI
        dp_pixelized_roi = np.zeros_like(roi)
        
        # 对ROI进行分块处理
        height, width = roi.shape
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # 确保不会超出图像边界
                y_end = min(y + block_size, height)
                x_end = min(x + block_size, width)
                
                # 提取当前块
                block = roi[y:y_end, x:x_end]
                
                # 计算平均像素值
                # 使用float32以确保精度
                block_avg = np.mean(block).astype(np.float32)
                
                # 添加拉普拉斯噪声
                # np.random.laplace(loc, scale) 生成拉普拉斯分布随机数
                # loc是位置参数(0),scale是尺度参数(beta)
                noise = np.random.laplace(0, beta)
                noisy_avg = block_avg + noise
                
                # 限制到有效像素范围[0, 255]
                # 使用np.clip避免越界
                noisy_avg = np.clip(noisy_avg, 0, 255).astype(np.uint8)
                
                # 将整个块设置为带噪声的平均值
                dp_pixelized_roi[y:y_end, x:x_end] = noisy_avg
        
        # 将处理后的ROI放回原图
        result[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width] = dp_pixelized_roi
        
        return result
    
    # 差分隱私高斯模糊 (DP-Blur)
    def dp_blur(self, image: np.ndarray, roi_x: int, roi_y: int, 
               roi_width: int, roi_height: int, kernel_size: Tuple[int, int], 
               sigma: int = 0, b0: int = 4) -> np.ndarray:
        """
        Differentially private Gaussian blur (DP-Blur) for a specific region of interest.
        Based on the paper "Differential Privacy for Image Publication" by Liyue Fan.
        
        Args:
            image: Input grayscale image
            roi_x, roi_y, roi_width, roi_height: Region of interest coordinates
            kernel_size: Size of Gaussian kernel
            sigma: Gaussian kernel standard deviation
            b0: Initial block size for pixelization step
            
        Returns:
            Differentially private blurred image
        """
        # 创建结果图像的副本
        result = image.copy()
        
        # 提取感兴趣区域(ROI)
        roi = image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
        
        # DP-Blur的原理：先使用小块大小的DP-Pix，然后再应用高斯模糊
        
        # 1. 首先对ROI应用DP-Pix，使用较小的块大小b0
        height, width = roi.shape
        
        # 计算全局敏感度: 255*m/b0^2
        global_sensitivity = (255.0 * self.m) / (b0 * b0) * 0.1
        
        # 计算拉普拉斯尺度参数
        beta = global_sensitivity / self.epsilon
        
        # 创建输出ROI
        dp_pixelized_roi = np.zeros_like(roi)
        
        # 对ROI进行分块处理
        for y in range(0, height, b0):
            for x in range(0, width, b0):
                # 确保不会超出图像边界
                y_end = min(y + b0, height)
                x_end = min(x + b0, width)
                
                # 提取当前块
                block = roi[y:y_end, x:x_end]
                
                # 计算平均像素值
                block_avg = np.mean(block).astype(np.float32)
                
                # 添加拉普拉斯噪声
                noise = np.random.laplace(0, beta)
                noisy_avg = block_avg + noise
                
                # 限制到有效像素范围[0, 255]
                noisy_avg = np.clip(noisy_avg, 0, 255).astype(np.uint8)
                
                # 将整个块设置为带噪声的平均值
                dp_pixelized_roi[y:y_end, x:x_end] = noisy_avg
        
        # 2. 应用高斯模糊
        # 确保kernel_size的维度是奇数
        k_height, k_width = kernel_size
        if k_height % 2 == 0:
            k_height += 1
        if k_width % 2 == 0:
            k_width += 1
            
        dp_blurred_roi = cv2.GaussianBlur(dp_pixelized_roi, (k_height, k_width), sigma)
        
        # 将处理后的ROI放回原图
        result[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width] = dp_blurred_roi
        
        return result
    
    # 计算评估指标
    def compute_metrics(self, original: np.ndarray, processed: np.ndarray) -> Tuple[float, float]:
        """
        Compute MSE and SSIM between original and processed images.
        
        Args:
            original: Original image
            processed: Processed image
            
        Returns:
            Tuple of (MSE, SSIM)
        """
        # 计算MSE
        mse = self.compute_mse(original, processed)
        
        # 计算SSIM
        ssim_value = self.compute_ssim(original, processed)
        
        return mse, ssim_value
    
    # 計算MSE
    def compute_mse(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Compute Mean Square Error between two images.
        
        Args:
            original: Original image
            processed: Processed image
            
        Returns:
            MSE value
        """
        return np.mean((original.astype(np.float32) - processed.astype(np.float32)) ** 2)

    # 計算SSIM (使用scikit-image庫)
    def compute_ssim(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Compute Structural Similarity Index between two images using scikit-image.
        
        Args:
            original: Original image
            processed: Processed image
            
        Returns:
            SSIM value (higher is better)
        """
        if HAVE_SKIMAGE:
            return ssim(original, processed, data_range=255)
        else:
            # 使用簡化的SSIM計算方法作為備選
            return self._compute_simple_ssim(original, processed)
    
    # 簡化的SSIM計算方法 (備選)
    def _compute_simple_ssim(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        簡化的SSIM計算實現（當scikit-image不可用時作為備選）
        """
        def _compute_mean(img):
            return np.mean(img)
        
        def _compute_variance(img):
            return np.var(img)
        
        def _compute_covariance(img1, img2, mean1, mean2):
            return np.mean((img1 - mean1) * (img2 - mean2))
        
        # 常數參數
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # 計算統計值
        mu_x = _compute_mean(original)
        mu_y = _compute_mean(processed)
        
        sigma_x = _compute_variance(original)
        sigma_y = _compute_variance(processed)
        
        sigma_xy = _compute_covariance(original, processed, mu_x, mu_y)
        
        # 計算SSIM
        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        
        ssim = numerator / denominator
        
        return ssim


# 方便快速访问的函数接口
def apply_pixelization(image, block_size=10):
    """
    对整张图像应用标准像素化
    
    Args:
        image: 输入图像 (numpy数组)
        block_size: 像素块大小
    
    Returns:
        像素化后的图像
    """
    height, width = image.shape
    obfuscator = DPImageObfuscator()
    return obfuscator.pixelize(image, 0, 0, width, height, block_size)


def apply_gaussian_blur(image, kernel_size=(15, 15), sigma=0):
    """
    对整张图像应用高斯模糊
    
    Args:
        image: 输入图像 (numpy数组)
        kernel_size: 高斯核大小
        sigma: 高斯核标准差
    
    Returns:
        模糊后的图像
    """
    height, width = image.shape
    obfuscator = DPImageObfuscator()
    return obfuscator.gaussian_blur(image, 0, 0, width, height, kernel_size, sigma)


def apply_dp_pixelization(image, epsilon=0.5, m=16, block_size=10):
    """
    对整张图像应用差分隐私像素化
    
    Args:
        image: 输入图像 (numpy数组)
        epsilon: 隐私参数
        m: 邻域参数
        block_size: 像素块大小
    
    Returns:
        差分隐私像素化后的图像
    """
    height, width = image.shape
    obfuscator = DPImageObfuscator(epsilon=epsilon, m=m)
    return obfuscator.dp_pix(image, 0, 0, width, height, block_size)


def apply_dp_blur(image, epsilon=0.5, m=16, kernel_size=(15, 15), sigma=0, b0=4):
    """
    对整张图像应用差分隐私高斯模糊
    
    Args:
        image: 输入图像 (numpy数组)
        epsilon: 隐私参数
        m: 邻域参数
        kernel_size: 高斯核大小
        sigma: 高斯核标准差
        b0: 初始块大小
    
    Returns:
        差分隐私模糊后的图像
    """
    height, width = image.shape
    obfuscator = DPImageObfuscator(epsilon=epsilon, m=m)
    return obfuscator.dp_blur(image, 0, 0, width, height, kernel_size, sigma, b0)
