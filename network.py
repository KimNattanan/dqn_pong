import torch
import torch.nn as nn
import torch.optim.adam

class Network(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(6,100)
    self.fc2 = nn.Linear(100,100)
    self.fc3 = nn.Linear(100,3)
    self.fc4 = nn.Linear(100,50)
    self.fc5 = nn.Linear(50,1)
    self.relu = nn.ReLU()

  def forward(self,x):
    x = self.relu(self.fc1(x)) # 100
    V = self.relu(self.fc4(x)) # 50
    V = self.fc5(V) # 1
    A = self.relu(self.fc2(x)) # 100
    A = self.fc3(A)
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