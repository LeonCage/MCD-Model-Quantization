
import torch
import torch.nn as nn
import torch.nn.functional as F

class MCDModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MCDModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        out = self.classifier(x)
        return out
