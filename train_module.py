''''

#for google colab
# 1. Mount Google Drive

from google.colab import drive

drive.mount('/content/gdrive')

# 2. Prepare data

!scp '/content/gdrive/My Drive/signdetection/data.zip' '/content/data.zip'

!unzip '/content/data.zip' -d '/content/'

# 3. Install Ultralytics

!pip install ultralytics

# 4. Train model

import os

!pip install ultralytics # Install ultralytics just before we need to use it

from ultralytics import YOLO


# Load a model
model = YOLO("yolov8n-cls.pt")  # load a pretained model

# Use the model
results = model.train(data='/content/data/', epochs=10)  # train the model

# 5. Copy results

!scp -r /content/runs '/content/gdrive/My Drive/signdetection'

'''