import cv2
import numpy as np
from skimage import filters, feature

class DifficultyScorer:
    def __init__(self):
        self.methods = {
            'blur': self._blur_score,
            'edges': self._edge_score,
            'entropy': self._entropy_score,
            'texture': self._texture_score
        }
    
    def compute_difficulty(self, image_tensor, weights=None):
        """Compute difficulty scores for a PyTorch tensor image"""
        # Convert tensor to numpy and permute dimensions
        image_np = image_tensor.numpy().transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)  # Scale to 0-255
        
        if weights is None:
            weights = {'blur': 0.4, 'edges': 0.3, 'entropy': 0.2, 'texture': 0.1}
        
        scores = {}
        total = 0.0
        for name, method in self.methods.items():
            score = method(image_np)
            scores[name] = score
            total += weights.get(name, 0) * score
        
        return total / sum(weights.values()), scores
    
    def _blur_score(self, image):
        """Compute blur metric (Laplacian variance)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _edge_score(self, image):
        """Compute edge density"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = filters.sobel(gray)
        return np.mean(edges)
    
    def _entropy_score(self, image):
        """Compute image entropy"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hist = np.histogram(gray, bins=256, range=(0, 255))[0]
        hist = hist / hist.sum()
        return -np.sum(hist * np.log2(hist + 1e-7))
    
    def _texture_score(self, image):
        """Compute texture complexity using LBP"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
        hist = np.histogram(lbp, bins=59, range=(0, 58))[0]
        hist = hist / hist.sum()
        return -np.sum(hist * np.log2(hist + 1e-7))