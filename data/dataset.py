from PIL import Image
from torch.utils.data import Dataset

class UAVDepthDataset(Dataset):
    def __init__(self, image_dir=None, transform=None):
        # Default to Colab path if not specified
        self.image_dir = image_dir or 'data/images'
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.image_dir) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)  # Should return tensor
        else:
            # Add default transform if none provided
            image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1)/255.0
        return image, self.image_files[idx]
