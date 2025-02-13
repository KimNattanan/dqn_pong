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
    self.optimizer = torch.optim.Adam(self.parameters(),lr=0.001)
    self.loss_func = nn.MSELoss()

  def forward(self,x):
    x = torch.tensor(x).to('cuda') # 6
    x = self.relu(self.fc1(x)) # 100
    V = self.relu(self.fc4(x)) # 50
    V = self.fc5(V) # 1
    A = self.relu(self.fc2(x)) # 100
    A = self.fc3(A)
    Q = torch.add(V,other=torch.mean(A),alpha=-1) + A
    return Q
  
  def fit(self,x,y):
    x = torch.tensor(x).to('cuda')
    y = torch.tensor(y).to('cuda')
    self.optimizer.zero_grad()
    pred = self(x)
    loss = self.loss_func(pred,y)
    loss.backward()
    self.optimizer.step()

  def save(self,path):
    torch.save(self,path)