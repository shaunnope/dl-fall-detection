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
    pbar = tqdm(train_loader)
    for i, (inputs, labels) in enumerate(pbar):
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      total_loss, loss = criterion(outputs, labels)
      total_loss.backward()
      optimizer.step()
      running_loss += loss
      if i % 10 == 9:
        lbox, lobj, lcls = running_loss / 10
        pbar.set_description(f"[{epoch + 1}, {i + 1}] train loss: {lbox:.4f}, {lobj:.4f}, {lcls:.4f}")
        # print(f"[{epoch + 1}, {i + 1}] train loss: {lbox:.4f}, {lobj:.4f}, {lcls:.4f}")
        running_loss = torch.zeros(3, device=device)

      if i % 20 == 19:
        torch.save(model.state_dict(), f"{save_dir}/model_{epoch}_{i}.pt")
      

    model.eval()
    running_loss = torch.zeros(3, device=device)
    pbar = tqdm(valid_loader)
    for i, (inputs, labels) in enumerate(pbar):
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs, True)
      _, loss = criterion(outputs, labels)
      running_loss += loss
      if i % 10 == 9:
        lbox, lobj, lcls = running_loss / i
        pbar.set_description(f"[{epoch + 1}, {i + 1}] valid loss: {lbox:.4f}, {lobj:.4f}, {lcls:.4f}")
    lbox, lobj, lcls = running_loss / len(valid_loader)
    print(f"[{epoch + 1}] valid loss: {lbox:.4f}, {lobj:.4f}, {lcls:.4f}")

  total_loss = torch.zeros(3, device=device)
  pbar = tqdm(test_loader)
  for i, (inputs, labels) in enumerate(pbar):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs, True)
    _, loss = criterion(outputs, labels)
    total_loss += loss
    if i % 10 == 9:
      lbox, lobj, lcls = total_loss / i
      pbar.set_description(f"[{epoch + 1}, {i + 1}] test loss: {lbox:.4f}, {lobj:.4f}, {lcls:.4f}")

  print(f"Final test loss: {total_loss / len(test_loader)}")

  print("Finished Training")