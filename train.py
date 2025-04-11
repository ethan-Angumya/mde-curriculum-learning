import os
import yaml
import torch
from torchvision import transforms
from data.dataset import UAVDepthDataset
from data.scoring import DifficultyScorer
from data.pacing import PacingFunction
from models.depth_estimation import DepthEstimationModel
from training.curriculum import CurriculumTrainer

# Colab-specific setup
if 'COLAB_GPU' in os.environ:
    os.environ['TORCH_HUB'] = '/content/torch_hub'  # Avoid repeated downloads
    from google.colab import drive
    drive.mount('/content/drive')

def main():
    # Debug: Verify CUDA and packages
    print("=== Environment Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load config
    try:
        with open('configs/model_params.yaml') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # Data pipeline with validation
    transform = transforms.Compose([
        transforms.Resize((384, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])
    
    try:
        dataset = UAVDepthDataset('data/images/', transform=transform)
        
        # Debug: Verify first sample
        sample_img, _ = dataset[0]
        print("\n=== Data Validation ===")
        print(f"Sample type: {type(sample_img)}")
        print(f"Sample shape: {sample_img.shape}")
        print(f"Value range: {sample_img.min():.3f} to {sample_img.max():.3f}")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # Initialize components
    try:
        model = DepthEstimationModel(config)
        scorer = DifficultyScorer()
        pacing_fn = PacingFunction().get_function('exponential')
    except Exception as e:
        print(f"Component initialization failed: {e}")
        return

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n=== Training Setup ===")
    print(f"Using device: {device.upper()}")
    print(f"Dataset size: {len(dataset)} images")
    print(f"Batch size: {config['training']['batch_size']}")
    print("Starting training...\n")

    # Train with error handling
    try:
        trainer = CurriculumTrainer(model, dataset, scorer, pacing_fn, device=device)
        losses = trainer.train(config['training'].get('num_iterations', 10000))
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
    finally:
        # Save model in Colab environment
        if 'COLAB_GPU' in os.environ:
            save_path = '/content/drive/MyDrive/Colab Data/mde-model.pth'
            torch.save(model.state_dict(), save_path)
            print(f"\nModel saved to: {save_path}")

if __name__ == '__main__':
    main()
