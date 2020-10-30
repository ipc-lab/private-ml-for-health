import io
from PIL import Image
from torchvision import models
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import urllib
import os

def get_model_from_global_agent():
    global_model = models.squeezenet1_1(pretrained=True)
    global_model.classifier[1] = nn.Conv2d(512, 5, kernel_size=(1,1), stride=(1,1))
    global_model.num_classes = 5
    global_model.to(torch.device('cpu'))
    map_location=torch.device('cpu')
    model_weights_link = 'https://drive.google.com/uc?id=11pb2yJKXgyYC9XnB9cd6HlNCFNxnlY1D'
    model_weights_path = './model/squeezenet_0.pt'
    urllib.request.urlretrieve(model_weights_link, model_weights_path)
    global_model.load_state_dict(torch.load("./model/squeezenet_0.pt", map_location=torch.device('cpu')))
    os.remove(model_weights_path)
    global_model.eval()
    return global_model


def transform_image(image_bytes):
    apply_transform = transforms.Compose([transforms.Resize(265),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return apply_transform(image).unsqueeze(0)



# change to DR dataset format
def format_class_name(class_name):
    class_name = class_name.replace('_', ' ')
    class_name = class_name.title()
    return class_name
