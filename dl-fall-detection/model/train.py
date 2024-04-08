import os
from tqdm import tqdm
import torch
import torch.optim as optim


def train(model, train_loader, valid_loader, test_loader, epochs, criterion, save_dir="runs"):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  
  device = next(model.parameters()).device
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  for epoch in range(epochs):
    model.train()
    running_loss = torch.zeros(3, device=device)
    for i, (inputs, labels) in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      total_loss, loss = criterion(outputs, labels)
      total_loss.backward()
      optimizer.step()
      running_loss += loss
      if i % 10 == 9:
        print(f"[{epoch + 1}, {i + 1}] train loss: {running_loss / 10}")
        running_loss = torch.zeros(3, device=device)

      if i % 20 == 19:
        torch.save(model.state_dict(), f"{save_dir}/model_{epoch}_{i}.pt")
      

    model.eval()
    for i, (inputs, labels) in tqdm(enumerate(valid_loader, 0), total=len(valid_loader)):
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      _, loss = criterion(outputs, labels)
      if i % 10 == 9:
        print(f"[{epoch + 1}, {i + 1}] valid loss: {loss}")

  for i, (inputs, labels) in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, loss = criterion(outputs, labels)
    if i % 10 == 9:
      print(f"[{epoch + 1}, {i + 1}] test loss: {loss}")

  print(f"Final test loss: {loss}")

  print("Finished Training")