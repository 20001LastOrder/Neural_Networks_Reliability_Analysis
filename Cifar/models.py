from torch import nn
import torch

class VotingNet(nn.Module):
    def __init__(self, models, output_dim):
        super().__init__()
        self.models = models
        self.output_dim = output_dim
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 
                                   'cpu')
        for model in self.models:
            model.eval()
    
    def forward(self, inputs):
        votings = torch.zeros(inputs.shape[0], self.output_dim).to(self.device)
        for model in self.models:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            # add 1 to the voting result
            votings[range(predicted.size(0)), predicted] += 1
        return votings

class ThresholdVotingNet(nn.Module):
    def __init__(self, models, output_dim, threshold):
        super().__init__()
        self.models = models
        self.output_dim = output_dim
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 
                                   'cpu')
        self.threshold = threshold
        for model in self.models:
            model.eval()
    
    def forward(self, inputs):
        votings = torch.zeros(inputs.shape[0], self.output_dim).to(self.device)
        for model in self.models:
            outputs = model(inputs)
            filtered = (outputs >= self.threshold).int() * outputs
            # add 1 to the voting result
            votings += filtered
        return votings