from torchvision import transforms
from PIL import Image, ImageFile, ImageFilter
import random

def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

def image_train(dataset, resize_size= 256, crop_size= 224):
    normalize = transforms.Normalize(mean= [0.485, 0.456, 0.406], 
                                    std= [0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), 
        transforms.RandomHorizontalFlip(), 
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3), 
        transforms.RandomGrayscale(), 
        transforms.ToTensor(), 
        normalize
    ])    

def img_test(dataset, resize= 256, crop_size= 224):
    normalize = transforms.Normalize(mean= [0.485, 0.456, 0.406], 
                                    std= [0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        normalize
    ])                     