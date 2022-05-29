#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.utils.data.dataloader as dl
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage import io, img_as_ubyte
from skimage.transform import resize
import time
import os
from resnet import ResNet18

TRAINING_DATA_PROCESSED_DIR = './_training_data_processed'
TESTING_DATA_PROCESSED_DIR = './_testing_data_processed'

RAW_TRAINING_DATA_DIR = './training_data'
RAW_TESTING_DATA_DIR = './testing_data'

IMAGE_INPUT_SIZE = 128

# Applies preprocessing to all images and saves them to disk.
def preprocess_images():
  def crop_image_to_content(img):
    top, left, bottom, right = (1e99, 1e99, 0, 0)
    for y in range(len(img)):
      for x in range(len(img[0])):
        pixel = img[y][x][0]
        if pixel < 1:
          top = min(y, top)
          left = min(x, left)
          bottom = max(y, bottom)
          right = max(x, right)
    return img[top:bottom, left:right]

  def preprocess_image(img):
    # Resize to speed up cropping
    img = resize(img, (128, 128), anti_aliasing=False)
    img = crop_image_to_content(img)
    img = resize(img, (64, 64), anti_aliasing=False)
    return img_as_ubyte(img)

  start_time = time.perf_counter()

  for class_folder in os.listdir(RAW_TESTING_DATA_DIR):
    for file in os.listdir(f'{RAW_TESTING_DATA_DIR}/{class_folder}'):
      input_path = f'{RAW_TESTING_DATA_DIR}/{class_folder}/{file}'
      input_img = io.imread(input_path)
      output_path = f'{TESTING_DATA_PROCESSED_DIR}/{class_folder}/{file}'
      output_img = preprocess_image(input_img)
      io.imsave(output_path, output_img)

  for class_folder in os.listdir(RAW_TRAINING_DATA_DIR):
    for file in os.listdir(f'{RAW_TRAINING_DATA_DIR}/{class_folder}'):
      input_path = f'{RAW_TRAINING_DATA_DIR}/{class_folder}/{file}'
      input_img = io.imread(input_path)
      output_path = f'{TRAINING_DATA_PROCESSED_DIR}/{class_folder}/{file}'
      output_img = preprocess_image(input_img)
      io.imsave(output_path, output_img)

  print(f'Preprocessed datasets in {time.perf_counter() - start_time}s.')

def run_network():
  torch.manual_seed(0)
  lr = 0.0001
  device = torch.device("cuda" if torch.cuda.is_available else "cpu")

  #model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg13_bn',pretrained=False).to(device)
  model = ResNet18(input_size=IMAGE_INPUT_SIZE, num_classes=10).to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  # For updating learning rate
  def update_lr(optimizer, lr):    
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
    
  transform = transforms.Compose([transforms.Resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE)), transforms.ToTensor()])

  training_dataset = torchvision.datasets.ImageFolder(TRAINING_DATA_PROCESSED_DIR, transform=transform)
  #training_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True, transform=transform)
  training_loader = dl.DataLoader(training_dataset, shuffle=True, num_workers=0, batch_size=20, pin_memory=True)

  testing_dataset = torchvision.datasets.ImageFolder(TESTING_DATA_PROCESSED_DIR, transform=transform)
  #testing_dataset = torchvision.datasets.MNIST('/files/', train=False, download=True, transform=transform)
  testing_loader = dl.DataLoader(testing_dataset, shuffle=True, num_workers=0, batch_size=20, pin_memory=True)

  cycle_errors = list[float]()
  test_accuracy = list[float]()

  num_epochs = 20
  for epoch in range(num_epochs):
    model.train(True)
    running_loss = 0
    if (epoch+1)%70 ==0:
      lr = lr/10
      update_lr(optimizer, lr)

    for (inputs, labels) in training_loader:
      optimizer.zero_grad(set_to_none=True)

      inputs, labels = inputs.to(device), labels.to(device)

      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    
    cycle_errors.append(running_loss)

    with torch.no_grad():
      model.train(False)
      correct = 0
      total = 0
      for (inputs, labels) in testing_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        #correct += (torch.argmax(outputs, dim=1) == labels).float().sum().item()
      acc = correct / total
      test_accuracy.append(acc)
      print("Epoch: [{}/{}], Training loss: {:.4f}, Accuracy: {}".format(epoch+1,num_epochs,running_loss, acc))

  plt.plot(range(1, len(cycle_errors) + 1), cycle_errors)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.show()

  plt.plot(range(1, len(test_accuracy) + 1), test_accuracy)
  plt.xlabel("Epoch")
  plt.ylabel("Test accuracy")
  plt.show()

if __name__ == '__main__':
  # Apply preprocessing. 
  # Takes a minute, so only run this if the logic for preprocessing is changed.
  #preprocess_images()
  run_network()
