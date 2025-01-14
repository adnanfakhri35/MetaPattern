import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
import argparse

class EncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # The exact structure will be determined by the state dict
            # This is just a placeholder until we load the state dict
        )
        
    def forward(self, x):
        return self.encoder(x)

def load_model_from_checkpoint(checkpoint_path):
    """Load model from checkpoint file"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = EncoderModel().to(device)
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], weights_only=True)
        elif 'model_state' in checkpoint:
            model_states = checkpoint['model_state']
            combined_state = {}
            for state_dict in model_states:
                combined_state.update(state_dict)
            model.load_state_dict(combined_state, weights_only=True)
        else:
            raise ValueError("Unknown checkpoint format")
            
        model.eval()
        return model
    
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess a single image"""
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    try:
        img = Image.open(image_path).convert('RGB')
        x = transform(img)
        x = x.unsqueeze(0)  # Add batch dimension
        return x
    except Exception as e:
        raise Exception(f"Error processing image {image_path}: {str(e)}")

def predict_spoof(model, image):
    """Make prediction using the loaded model"""
    try:
        device = next(model.parameters()).device
        image = image.to(device)
        
        with torch.no_grad():
            prediction = model(image)
            
            # If model returns multiple outputs, take the first one
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            
            # Apply softmax if needed
            if len(prediction.shape) > 1:
                prediction = torch.softmax(prediction, dim=1)
            
            return prediction.cpu().numpy()
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")

def process_directory(model, directory_path, supported_formats=('.jpg', '.jpeg', '.png', '.bmp')):
    """Process all images in a directory"""
    results = {}
    directory = Path(directory_path)
    
    for image_path in directory.rglob("*"):
        if image_path.suffix.lower() in supported_formats:
            try:
                # Preprocess image
                image_tensor = preprocess_image(str(image_path))
                
                # Make prediction
                prediction = predict_spoof(model, image_tensor)
                
                # Store results
                relative_path = str(image_path.relative_to(directory))
                results[relative_path] = {
                    'prediction': prediction.tolist(),
                    'score': float(prediction.max())
                }
                
                print(f"Processed {relative_path}: Score = {results[relative_path]['score']:.4f}")
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test images with checkpoint model')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--input', required=True, help='Path to input image or directory')
    parser.add_argument('--output', default='results.txt', help='Path to output results file')
    args = parser.parse_args()
    
    try:
        # Load model
        print("Loading model...")
        model = load_model_from_checkpoint(args.checkpoint)
        print("Model loaded successfully!")
        
        # Process input
        input_path = Path(args.input)
        if input_path.is_file():
            # Single image
            image_tensor = preprocess_image(str(input_path))
            prediction = predict_spoof(model, image_tensor)
            results = {str(input_path): {
                'prediction': prediction.tolist(),
                'score': float(prediction.max())
            }}
        else:
            # Directory
            results = process_directory(model, input_path)
        
        # Save results
        with open(args.output, 'w') as f:
            for path, result in results.items():
                f.write(f"Image: {path}\n")
                f.write(f"Prediction: {result['prediction']}\n")
                f.write(f"Score: {result['score']:.4f}\n")
                f.write("-" * 50 + "\n")
        
        print(f"\nResults saved to {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
