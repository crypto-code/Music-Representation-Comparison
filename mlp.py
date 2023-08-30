import torch.nn as nn
import torch.nn.functional as F


class MERT_MLP(nn.Module):
    def __init__(
            self,
            num_features,
            hidden_layer_sizes,
            num_outputs,
            dropout_input=True,
            dropout_p=0.5,
    ):
        super().__init__()
        d = num_features
        self.aggregator = nn.Conv1d(in_channels=25, out_channels=1, kernel_size=1)
        self.num_layers = len(hidden_layer_sizes)
        for i, ld in enumerate(hidden_layer_sizes):
            setattr(self, f"hidden_{i}", nn.Linear(d, ld))
            d = ld
        self.output = nn.Linear(d, num_outputs)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, y):
        x = self.aggregator(x).squeeze()
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = getattr(self, f"hidden_{i}")(x)
            x = F.relu(x)
            x = self.dropout(x)
        logits = self.output(x)
        return logits, F.binary_cross_entropy_with_logits(logits, y.float(), reduction="mean")

class SimpleMLP(nn.Module):
    def __init__(
            self,
            num_features,
            hidden_layer_sizes,
            num_outputs,
            dropout_input=True,
            dropout_p=0.5,
    ):
        super().__init__()
        d = num_features
        self.num_layers = len(hidden_layer_sizes)
        for i, ld in enumerate(hidden_layer_sizes):
            setattr(self, f"hidden_{i}", nn.Linear(d, ld))
            d = ld
        self.output = nn.Linear(d, num_outputs)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, y):
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = getattr(self, f"hidden_{i}")(x)
            x = F.relu(x)
            x = self.dropout(x)
        logits = self.output(x)
        return logits, F.binary_cross_entropy_with_logits(logits, y.float(), reduction="mean")
