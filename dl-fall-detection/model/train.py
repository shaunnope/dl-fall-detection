import os
from tqdm import tqdm
import torch
import torch.optim as optim

from torcheval.metrics.functional import multiclass_f1_score as f1, multiclass_precision_recall_curve as prc
import matplotlib.pyplot as plt

from eval.display import plot_prc, plot_losses

def train(model, train_loader, valid_loader, test_loader, epochs, criterion, save_dir="runs"):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  
  device = next(model.parameters()).device
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  tlboxes = []
  tlobjs = []
  tlclss = []
  train_iters = []

  vlboxes = []
  vlobjs = []
  vlclss = []
  vf1 = []
  valid_iters = []
  
  best = {}
  for t in ["train", "valid"]:
    best[t] = (float("inf"), 0, 0)
  
  try:
    nc = model.head.nc
    for epoch in range(epochs):
      model.train()
      running_loss = torch.zeros(3, device=device)
      n = len(train_loader)
      pbar = tqdm(train_loader)
      for i, (inputs, labels) in enumerate(pbar):
        torch.cuda.empty_cache()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        total_loss, loss = criterion(outputs, labels)
        total_loss.backward()
        optimizer.step()

        del inputs, labels, outputs

        running_loss += loss
        if i % 4 == 3:
          lbox, lobj, lcls = running_loss / 4
          tlboxes.append(lbox.item())
          tlobjs.append(lobj.item())
          tlclss.append(lcls.item())
          train_iters.append(epoch * n + i)

          pbar.set_description(f"[{epoch + 1}, {i + 1}] train: {lbox:.4f}, {lobj:.4f}, {lcls:.4f}")
          running_loss = torch.zeros(3, device=device)

        if total_loss < best["train"][0]:
          torch.save(model.state_dict(), f"{save_dir}/model_{epoch}_best_train.pt")
          best["train"] = (total_loss.item(), epoch, i)

        if i % 20 == 19:
          torch.save(model.state_dict(), f"{save_dir}/model_{epoch}_{i}.pt")

      del total_loss, loss, running_loss
        
      model.eval()
      running_loss = torch.zeros(3, device=device)
      total_loss = 0.0

      vpreds = torch.tensor([], device=device, dtype=torch.int64)
      vtargets = torch.tensor([], device=device, dtype=torch.int64)

      n = len(valid_loader)
      pbar = tqdm(valid_loader)
      for i, (inputs, labels) in enumerate(pbar):
        torch.cuda.empty_cache()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs, True)
        ttl, loss, preds, targets = criterion(outputs, labels, True)

        vpreds = torch.cat([vpreds, preds], 0)
        vtargets = torch.cat([vtargets, targets], 0)

        del inputs, labels, outputs, preds, targets
        running_loss += loss
        total_loss += ttl

        if i % 4 == 3:
          lbox, lobj, lcls = running_loss / i
          vlboxes.append(lbox.item())
          vlobjs.append(lobj.item())
          vlclss.append(lcls.item())
          valid_iters.append(epoch * n + i)

          f1_scores = f1(vpreds, vtargets, num_classes = nc, average=None)
          f1s = " ".join([f"{r:.3f}" for r in f1_scores.tolist()])
          vf1.append(f1_scores.tolist())

          del f1_scores, vpreds, vtargets, lbox, lobj, lcls
          vpreds = torch.tensor([], device=device, dtype=torch.int64)
          vtargets = torch.tensor([], device=device, dtype=torch.int64)

          pbar.set_description(f"[{epoch + 1}, {i + 1}] valid: {lbox:.4f}, {lobj:.4f}, {lcls:.4f}, f1: {f1s}")
      
      lbox, lobj, lcls = running_loss / len(valid_loader)
      vlboxes.append(lbox.item())
      vlobjs.append(lobj.item())
      vlclss.append(lcls.item())
      valid_iters.append(epoch * n + i)

      f1_scores = f1(vpreds, vtargets, num_classes = nc, average=None)
      f1s = " ".join([f"{r:.3f}" for r in f1_scores.tolist()])
      # vf1.append(f1_scores.tolist())

      print(f"[{epoch + 1}] valid: {lbox:.4f}, {lobj:.4f}, {lcls:.4f}, f1: {f1s}")

      if total_loss < best["valid"][0]:
        torch.save(model.state_dict(), f"{save_dir}/model_best_val.pt")
        best["valid"] = (total_loss.item(), epoch, i)

      del total_loss, loss, running_loss, vpreds, vtargets, f1_scores

    total_loss = torch.zeros(3, device=device)

    tpreds = torch.tensor([], device=device, dtype=torch.int64)
    ttargets = torch.tensor([], device=device, dtype=torch.int64)

    n = len(test_loader)
    pbar = tqdm(test_loader)
    for i, (inputs, labels) in enumerate(pbar):
      torch.cuda.empty_cache()
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs, True)
      _, loss, preds, targets = criterion(outputs, labels, True)

      tpreds = torch.cat([tpreds, preds], 0)
      ttargets = torch.cat([ttargets, targets], 0)

      del inputs, labels, outputs, preds, targets
      total_loss += loss

      if i % 4 == 3:
        lbox, lobj, lcls = total_loss / i
        pbar.set_description(f"[{epoch + 1}, {i + 1}] test loss: {lbox:.4f}, {lobj:.4f}, {lcls:.4f}")      

    f1_scores = f1(tpreds, ttargets, num_classes = nc, average=None)
    f1s = " ".join([f"{r:.3f}" for r in f1_scores.tolist()])

    plot_prc(*prc(tpreds, ttargets, num_classes = nc), save_dir)

    print(f"Final test loss: {total_loss / len(test_loader)}, f1: {f1s}")

    print("Finished Training")

  except KeyboardInterrupt:
    print("Training interrupted")

    total_loss = torch.zeros(3, device=device)

  torch.save(model.state_dict(), f"{save_dir}/model_final.pt")

  return {
    "train": {
      "box": tlboxes,
      "cls": tlobjs,
      "dfl": tlclss,
      "iter": train_iters
    },
    "valid": {
      "box": vlboxes,
      "cls": vlobjs,
      "dfl": vlclss,
      "iter": valid_iters,
      "f1": vf1
    },
    "test": {
      "box": total_loss[0].item(),
      "cls": total_loss[1].item(),
      "dfl": total_loss[2].item()
    }
  }