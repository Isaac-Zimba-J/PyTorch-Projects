import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 for the default camera, or change it to 1, 2, etc. if you have multiple cameras

class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 5)
        
        self.fc1 = nn.Linear(256, 2)
        
        self.pool = nn.MaxPool2d(2,2)
        
    def forward(self, data):
        data = self.pool(F.relu(self.conv1(data)))
        data = self.pool(F.relu(self.conv2(data)))
        data = self.pool(F.relu(self.conv3(data)))
        data = self.pool(F.relu(self.conv4(data)))
        
        bs, _, _, _ = data.shape
        
        data = F.adaptive_avg_pool2d(data, 1).reshape(bs, -1)
        data = self.fc1(data)
        
        return data

# Load the pre-trained model
labels = ['daisy','dandelion']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvModel().to(device)
checkpoint = torch.load('model_outputs/model_flowers.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

while True:
    ret, frame = cap.read()

    # Perform inference on each frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = transform(image)
    image = torch.unsqueeze(image, 0)

    with torch.no_grad():
        outputs = model(image.to(device))

    outputs_label = torch.topk(outputs, 1)
    
    pred_class = labels[int(outputs_label.indices)]
    print(pred_class)

    cv2.putText(frame, f"Prediction: {pred_class}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Object Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
