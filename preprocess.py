import numpy as np
import os
import cv2
from tqdm import tqdm

class ConvolutionalFeatureExtractor:
    def __init__(self, kernel_size=3, target_pool_size=8):
        self.kernel_size = kernel_size
        self.target_pool_size = target_pool_size
    
    # ----------------------------------------------------------
    # 1. Convolution filter bank
    # ----------------------------------------------------------
    def apply_conv(self, image, grayscale=False, kernel_size=3):
        if grayscale:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            channels = [gray.astype(np.float32)]
        else:
            channels = [image[..., c].astype(np.float32) for c in range(image.shape[2])]
            
        filters = []

        for ch in channels:
            sobelx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=kernel_size)
            sobely = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=kernel_size)
            filters.extend([sobelx, sobely])

            lap = cv2.Laplacian(ch, cv2.CV_32F)
            filters.append(lap)

            for theta in np.arange(0, np.pi, np.pi / 6):
                kernel = cv2.getGaborKernel((11, 11), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                fimg = cv2.filter2D(ch, cv2.CV_32F, kernel)
                filters.append(fimg)

            g1 = cv2.GaussianBlur(ch, (3, 3), 1)
            g2 = cv2.GaussianBlur(ch, (5, 5), 2)
            dog = g1 - g2
            filters.append(dog)

        stacked = np.stack(filters, axis=-1).astype(np.float32)

        stacked = np.moveaxis(stacked, -1, 0)
        normed = []
        for ch in stacked:
            mn, mx = ch.min(), ch.max()
            if mx - mn < 1e-6:
                normed.append(np.zeros_like(ch))
            else:
                normed.append((ch - mn) / (mx - mn))
        return np.moveaxis(np.array(normed, dtype=np.float32), 0, -1)

    # ----------------------------------------------------------
    # 2. Max pooling
    # ----------------------------------------------------------
    def apply_maxpooling(self, image, target_pool_size=8):
        pooled = image.copy().astype(np.float32)

        def safe_resize(img, new_w, new_h):
            if img.ndim == 2 or img.shape[2] <= 4:
                return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                return np.stack(
                    [cv2.resize(img[..., c], (new_w, new_h), interpolation=cv2.INTER_AREA)
                     for c in range(img.shape[2])],
                    axis=-1
                )

        while True:
            h, w = pooled.shape[:2]
            if h <= target_pool_size and w <= target_pool_size:
                break
            pooled = safe_resize(pooled, max(1, w // 2), max(1, h // 2))

        if pooled.shape[0] != target_pool_size or pooled.shape[1] != target_pool_size:
            pooled = safe_resize(pooled, target_pool_size, target_pool_size)

        return pooled

    # ----------------------------------------------------------
    # 3. Full feature extraction
    # ----------------------------------------------------------
    def feature_extraction(self, image, grayscale=False, kernel_size=3, target_pool_size=7):
        conv_output = self.apply_conv(image, grayscale=grayscale, kernel_size=kernel_size)
        pooled_output = self.apply_maxpooling(conv_output, target_pool_size=target_pool_size)

        features = pooled_output.flatten()
        means = pooled_output.mean(axis=(0, 1))
        stds = pooled_output.std(axis=(0, 1))

        return np.concatenate([features, means, stds])

    # ----------------------------------------------------------
    # 4. Batch feature extraction
    # ----------------------------------------------------------
    def batch_CFE(self, batch_size, base_dir, grayscale=False, kernel_size=3, target_size=7):
        processed = []
        img_dir = os.listdir(base_dir)

        for i in tqdm(range(0, len(img_dir), batch_size)):
            batch_files = img_dir[i:i + batch_size]

            for filename in batch_files:
                path = os.path.join(base_dir, filename)
                img = cv2.imread(path)
                if img is None:
                    continue

                feat = self.feature_extraction(img, grayscale=grayscale,
                                               kernel_size=kernel_size,
                                               target_pool_size=target_size)
                processed.append(feat)

        return np.array(processed)

# ----------------------------------------------------------
# RUN TEST
# ----------------------------------------------------------
if __name__ == "__main__":
    base_dir = "E:/@IIT_BBS/@Sem 1/ML/Final Project/Safety-Analysis/Dataset/Train_Images"
    extractor = ConvolutionalFeatureExtractor()
    print(extractor)
