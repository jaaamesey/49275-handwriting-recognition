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

TRAINING_DATA_PROCESSED_DIR = './_training_data_processed'
TESTING_DATA_PROCESSED_DIR = './_testing_data_processed'

RAW_TRAINING_DATA_DIR = './training_data'
RAW_TESTING_DATA_DIR = './testing_data'

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
  device = torch.device("cuda" if torch.cuda.is_available else "cpu")

  model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11').to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

  transform = transforms.ToTensor()

  training_dataset = torchvision.datasets.ImageFolder(TRAINING_DATA_PROCESSED_DIR, transform=transform)
  training_loader = dl.DataLoader(training_dataset, shuffle=True, num_workers=0, batch_size=256, pin_memory=True)

  testing_dataset = torchvision.datasets.ImageFolder(TESTING_DATA_PROCESSED_DIR, transform=transform)
  testing_loader = dl.DataLoader(testing_dataset, shuffle=True, num_workers=0, batch_size=1, pin_memory=True)

  cycle_errors = list[float]()
  test_accuracy = list[float]()

  for epoch in range(100):
    model.train(True)
    running_loss = 0

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
      for (inputs, labels) in testing_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        correct += (torch.argmax(outputs, dim=1) == labels).float().sum().item()
      test_accuracy.append(correct / len(testing_loader))
      print(f'{correct} / {len(testing_loader)} ({correct / len(testing_loader)})')
      
    print(f'{epoch}: {running_loss}')

  plt.plot(range(1, len(cycle_errors) + 1), cycle_errors)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.show()

  plt.plot(range(1, len(test_accuracy) + 1), test_accuracy)
  plt.xlabel("Epoch")
  plt.ylabel("Test accuracy")
  plt.show()

  torch.save(model.state_dict(), './model')

if __name__ == '__main__':
  # Apply preprocessing. 
  # Takes a minute, so only run this if the logic for preprocessing is changed.
  #preprocess_images()
  run_network()
