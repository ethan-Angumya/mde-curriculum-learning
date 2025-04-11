import numpy as np
from tqdm import tqdm
import torch
from torch import nn

class CurriculumTrainer:
    def __init__(self, model, dataset, scorer, pacing_fn, device=None):
        
        """
        Args:
            model: The depth estimation model
            dataset: UAV image dataset
            scorer: Difficulty scorer instance
            pacing_fn: Pacing function for curriculum
            device: Training device (cuda/cpu)
        """
        self.dataset = dataset
        self.scorer = scorer
        self.pacing_fn = pacing_fn
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # Pre-compute difficulties for all images
        self.difficulties = self._compute_difficulties()
        self.sorted_indices = np.argsort(self.difficulties)
        
        # Training setup
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        self.criterion = nn.L1Loss()  # Using L1 loss for depth estimation

    def _compute_difficulties(self):
        """Compute difficulty scores for entire dataset"""
        difficulties = []
        for img, _ in tqdm(self.dataset, desc="Scoring images"):
            score, _ = self.scorer.compute_difficulty(np.array(img))
            difficulties.append(score)
        return np.array(difficulties)
    
    def get_curriculum_batch(self, iteration, batch_size=8):
        """Select batch according to current curriculum stage"""
        threshold = self.pacing_fn(iteration)
        max_idx = int(threshold * len(self.sorted_indices))
        eligible_indices = self.sorted_indices[:max_idx]
        
        batch_indices = np.random.choice(eligible_indices, size=batch_size, replace=False)
        batch = [self.dataset[i] for i in batch_indices]
        images = torch.stack([x[0] for x in batch]).to(self.device)
        return images
    
    def train_step(self, iteration):
        """Execute one training step"""
        images = self.get_curriculum_batch(iteration)
        
        self.optimizer.zero_grad()
        pred_depth = self.model(images)
        
        # Self-supervised loss (e.g., image reconstruction)
        loss = self.criterion(pred_depth, self._pseudo_ground_truth(images))
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _pseudo_ground_truth(self, images):
        """Generate proxy targets for self-supervised learning"""
        # Placeholder - in practice would use edge maps or other self-supervised signals
        return torch.rand_like(self.model(images))
    
    def train(self, num_iterations):
        """Run full training loop"""
        losses = []
        for i in tqdm(range(num_iterations), desc="Training"):
            loss = self.train_step(i)
            losses.append(loss)
            
            if i % 100 == 0:
                tqdm.write(f"Iter {i:05d} | Loss: {loss:.4f} | "
                          f"Curriculum: {self.pacing_fn(i):.2f}")
        return losses
