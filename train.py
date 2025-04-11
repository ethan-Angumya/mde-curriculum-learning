# Add this at the top of your train.py
import os
if 'COLAB_GPU' in os.environ:
    os.environ['TORCH_HUB'] = '/content/torch_hub'  # Avoid repeated downloads
    
import yaml
import torch  # This was missing
from torchvision import transforms
from data.dataset import UAVDepthDataset
from data.scoring import DifficultyScorer
from data.pacing import PacingFunction
from models.depth_estimation import DepthEstimationModel
from training.curriculum import CurriculumTrainer

def main():
    # Load config
    with open('configs/model_params.yaml') as f:
        config = yaml.safe_load(f)

    # Data pipeline
    transform = transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToTensor(),  # This converts PIL Image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    dataset = UAVDepthDataset('data/images/', transform=transform)

    # Initialize components
    model = DepthEstimationModel(config)
    scorer = DifficultyScorer()
    pacing_fn = PacingFunction().get_function('exponential')

    # Device detection with proper torch import
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device.upper()}\n")

    # Train
    trainer = CurriculumTrainer(model, dataset, scorer, pacing_fn, device=device)
    losses = trainer.train(10000)

if __name__ == '__main__':
    main()
