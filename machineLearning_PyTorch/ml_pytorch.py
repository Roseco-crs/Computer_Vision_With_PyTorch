import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import ToTensor
from torchvision import datasets

# 1. Download data from MNIST

training_data = datasets.MNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.MNIST(
    root= "data",
    train= False,
    download = True,
    transform=ToTensor(),
)

batch_size = 64

# create a loader to iterate over data
train_dataloader = DataLoader(training_data, batch_size= batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Define AI model

## Get device for training
device = torch.device(
"cuda" if toch.cuda is_available()
else "mps" if torch.backends.mps.is_available()
else "cpu"
)

## Define model
class NeuraNetwork(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super().__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, num_class)
    
  def forward(self, image_tensor):
    image_tensor = self.flatten(image_tensor)
    logits = self.linear_relu_stack(image_tensor)
    return logits
    
input_size = 28*28
hidden_size = 512
num_class =10
 
model = NeuralNetwork(input_size, hidden_size, num_classes)
 
# Train loop

## Define learing rate, loss function, optimizer
learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

## Define training function
def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train()
  
  for batch_num, (X,y) in enumerate(dataloader):
    X = X.to(device)
    y = y.to(device)
    
    # forward to compute prediction
    pred = model(X)
    # compute prediction error 
    loss = loss_fn(pred, y)
    
    # Backward 
    optimizer.zero_grad()	# zero any previous gradient
    loss.backward() 		# calculate gradient
    optimizer.step()		# update model parameters
    
    if batch_num >0 and batch_num % 100 ==0:
      loss = loss.item()
      current = batch_num*len(X)
      print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")
      
# Test loop

def test(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  model.eval()
  test_loss = 0
  correct = 0
  for X, y in dataloader:
    X = X.to(device)
    y = y.to(device)
    pred = model(X)
    test_loss += loss_fn(pred, y).item()
    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  test_loss /= num_batches
  correct /= size
  print(f"Test error: \n Accuracy: {100*correct:>0.1f}%, Avg loss: {test_loss:>8f}\n")
  
  
# Train model

epochs = 5
for e in range(epochs):
  print(f"Epoch {e+1} \n-----------------"}
  train(train_dataloader, model, loss_fn, optimizer)
  test(test_dataloader, model, loss_fn)
print("Done!")


# Save the model and make predictions

## Save our model parameters
torch.save(model.state_dict(), "ml_pytorh_model.pth")
print("Saved PyTorch Model State to ml_pytorch_model.pth")

## Load the saved model parameters into new instance of the model
model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)
model.load_state_dict(torch.load("ml_pytorch_model.pth"))

## inference using the new model instance
model.eval()
for i in range(10):
  x = test_data[i][0]
  y = test_data[i][1]
  x = x.to(device)
  pred = model(x)
  predicted = pred[0].argmax(0).item()
  actual = y
  print("Predicted: '{predicted}', Actual: '{actual}' ")



 