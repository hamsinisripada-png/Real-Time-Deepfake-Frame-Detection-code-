import torch
from PIL import Image
from torchvision import transforms

from model_loader import load_model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image_path):
    model = load_model()

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    probs = torch.nn.functional.softmax(output, dim=1)

    confidence, prediction = torch.max(probs, dim=1)

    print(f"Prediction Class: {prediction.item()}")
    print(f"Confidence: {confidence.item():.4f}")

if __name__ == "__main__":
    predict("sample_images/test.jpg")
