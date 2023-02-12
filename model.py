import torch.nn as nn


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=49, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=2),
            nn.ReLU(),
        )

    def forward(self, input):
        return self.model(input)
