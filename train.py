import os
import yaml
import torch
from torchvision import transforms

def check_colab():
    return 'COLAB_GPU' in os.environ

def main():
    # 1. Environment setup
    print("=== Environment Check ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    
    # 2. Late imports (after GPU check)
    from data.dataset import UAVDepthDataset
    from data.scoring import DifficultyScorer
    from data.pacing import PacingFunction
    from models.depth_estimation import DepthEstimationModel
    from training.curriculum import CurriculumTrainer

    # 3. Load config
    with open('configs/model_params.yaml') as f:
        config = yaml.safe_load(f)

    # 4. Training execution
    transform = transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = UAVDepthDataset('data/images/', transform=transform)
    model = DepthEstimationModel(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = CurriculumTrainer(
        model=model,
        dataset=dataset,
        scorer=DifficultyScorer(),
        pacing_fn=PacingFunction().get_function('exponential'),
        device=device
    )
    
    trainer.train(config.get('num_iterations', 10000))

if __name__ == '__main__':
    main()
