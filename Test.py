import cv2 as cv
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from Model import DeePixBiS
from Loss import PixWiseBCELoss
from Metrics import predict, test_accuracy, test_loss

model = DeePixBiS()
model.load_state_dict(torch.load('./DeePixBiS.pth'))
model.eval()

tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

faceClassifier = cv.CascadeClassifier('Classifiers/haarface.xml')

camera = cv.VideoCapture(0)

while cv.waitKey(1) & 0xFF != ord('q'):
    _, img = camera.read()
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)

    for x, y, w, h in faces:
        faceRegion = img[y:y + h, x:x + w]
        faceRegion = cv.cvtColor(faceRegion, cv.COLOR_BGR2RGB)
        # cv.imshow('Test', faceRegion)

        faceRegion = tfms(faceRegion)
        faceRegion = faceRegion.unsqueeze(0)

        mask, binary = model.forward(faceRegion)
        res = torch.mean(mask).item()
        # res = binary.item()
        print(res)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        if res < 0.5:
            cv.putText(img, 'Fake', (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        else:
            cv.putText(img, 'Real', (x, y + h + 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

    cv.imshow('Deep Pixel-wise Binary Supervision Anti-Spoofing', img)
