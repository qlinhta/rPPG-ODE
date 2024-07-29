import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import argparse
import warnings
import timm

warnings.filterwarnings("ignore")


def load_model(model_name, device):
    torchvision_models = [
        'resnet18', 'resnet50', 'densenet121', 'densenet201', 'mobilenet_v2', 'shufflenet_v2_x1_0',
        'squeezenet1_0', 'vgg16', 'vgg19', 'inception_v3', 'googlenet', 'resnext50_32x4d',
        'resnext101_32x8d', 'mnasnet1_0'
    ]
    timm_models = ['efficientnet_b0', 'efficientnet_b4', 'vit_base_patch16_224', 'vit_large_patch16_224',
                   'nasnetamobile', 'nasnetalarge']

    if model_name in torchvision_models:
        model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    elif model_name in timm_models:
        model = timm.create_model(model_name, pretrained=True)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    model.to(device)
    model.eval()
    return model


def extract_features_from_frame(frame, model, transform, device):
    image = Image.fromarray(frame).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.forward_features(image) if hasattr(model, 'forward_features') else model(image)
    return features.cpu().numpy().flatten()


def process_npy_files(input_dir, output_dir, model, transform, device, model_name):
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    count_files = len([file for file in os.listdir(input_dir) if file.endswith('_frames.npy')])
    progress = tqdm([file for file in os.listdir(input_dir) if file.endswith('_frames.npy')], total=count_files,
                    desc="Processing", unit="instance")
    for subject_file in progress:
        subject = subject_file.replace('_frames.npy', '')
        frames_path = os.path.join(input_dir, subject_file)
        frames = np.load(frames_path)
        features_list = []
        for frame in frames:
            features = extract_features_from_frame(frame, model, transform, device)
            features_list.append(features)
        features_array = np.array(features_list)
        np.save(os.path.join(model_dir, f"{subject}_features.npy"), features_array)
        progress.set_description(f"Processed {subject}")


def main(input_dir, output_dir, model_name):
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model = load_model(model_name, device)
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    process_npy_files(input_dir, output_dir, model, transform, device, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature extraction from video frames using a pretrained model.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the .npy files with frames.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where the extracted features will be saved.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the pretrained model to use (e.g., resnet18, swin_tiny_patch4_window7_224).")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.model_name)
