import torch
import torch.nn as nn
import torch.optim.adam

class Network(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(8,512)
    self.fc2 = nn.Linear(512,256)
    self.fc3 = nn.Linear(256,1)
    self.fc4 = nn.Linear(256,3)
    self.relu = nn.ReLU()

  def forward(self,x):
    x = self.relu(self.fc1(x)) # 512
    x = self.relu(self.fc2(x)) # 256
    V = self.fc3(x) # 1
    A = self.fc4(x) # 3
    Q = torch.add(V,other=torch.mean(A),alpha=-1) + A
    return Q
  
  def fit(self,x_batch,y_batch,optimizer,loss_func):
    optimizer.zero_grad()
    preds = self(x_batch)
    loss = loss_func(preds,y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()

  def save(self,path):
    torch.save(self,path)