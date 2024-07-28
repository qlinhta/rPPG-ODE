import torch
import torchvision.transforms as transforms
from PIL import Image
import timm

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(model_name):
    model = timm.create_model(model_name, pretrained=True)
    print(f"Loaded model: {model_name}")
    model.to(device)
    model.eval()
    return model


def extract_features(frames, model):
    features_list = []
    with torch.no_grad():
        for frame in frames:
            image = transform(Image.fromarray(frame.numpy().transpose(1, 2, 0)).convert('RGB')).unsqueeze(0).to(device)
            features = model.forward_features(image)
            features_list.append(features.squeeze().cpu().numpy())
    return torch.tensor(features_list, dtype=torch.float32)
