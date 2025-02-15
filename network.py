import torch
import torch.nn as nn
import torch.optim.adam

class Network(nn.Module):
  def __init__(self, input_dim=8, action_dim=3):
    super().__init__()
    self.fc1 = nn.Linear(input_dim,128)
    self.fc2 = nn.Linear(128,128)
    self.fc_value = nn.Linear(128,64)
    self.value = nn.Linear(64,1)
    self.fc_advantage = nn.Linear(128,64)
    self.advantage = nn.Linear(64,action_dim)

  def forward(self,x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    V = torch.relu(self.fc_value(x))
    V = self.value(V)
    A = torch.relu(self.fc_advantage(x))
    A = self.advantage(A)
    Q = V + (A - A.mean(dim=-1, keepdim=True))
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