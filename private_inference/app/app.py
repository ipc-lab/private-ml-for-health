import os
import matplotlib.image as mpimg
from flask import Flask, render_template, request, redirect
from PIL import Image
import io
from inference import get_prediction
from commons import format_class_name, transform_image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
import pickle
import time
import json

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
           return
        img_bytes = file.read()
        file_tensor = transform_image(image_bytes=img_bytes) #######
        class_name = get_prediction(file_tensor)
        return render_template('result.html', class_name=class_name)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
