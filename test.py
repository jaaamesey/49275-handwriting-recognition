import torch
import torch.utils.data.dataloader as dl
import torchvision.transforms as transforms
import torchvision

TRAINING_DATA_PROCESSED_DIR = './_training_data_processed'
TESTING_DATA_PROCESSED_DIR = './_testing_data_processed'

RAW_TRAINING_DATA_DIR = './training_data'
RAW_TESTING_DATA_DIR = './testing_data'

NUM_CLASSES = 10
IMAGE_INPUT_SIZE = 128

model = torch.jit.load('./handwriting993.pt')
transform = transforms.Compose([transforms.Resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE)), transforms.ToTensor()])
testing_dataset = torchvision.datasets.ImageFolder(TESTING_DATA_PROCESSED_DIR, transform=transform)
testing_loader = dl.DataLoader(testing_dataset, shuffle=True, num_workers=0, batch_size=1, pin_memory=True)

if __name__ == '__main__':
  torch.manual_seed(0)
  with torch.no_grad():
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    model.train(False)
    correct = 0
    total = 0
    for (inputs, labels) in testing_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      correct += (predicted == labels).sum().item()
      total += labels.size(0)
    acc = correct / total
    print("Accuracy: {}".format(acc))
