import torch
import torch.nn as nn

class TransformerGRUModel(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64,
        nhead=4,
        num_transformer_layers=2,
        gru_hidden_dim=64,
        num_classes=1,
        dropout=0.3
    ):
        super().__init__()

        print("\n[MODEL INIT]")
        print("Input dimension:", input_dim)

        self.input_projection = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden_dim,
            batch_first=True
        )

        self.norm = nn.LayerNorm(gru_hidden_dim)
        self.fc = nn.Linear(gru_hidden_dim, num_classes)

        print("[MODEL INIT COMPLETED]")

    def forward(self, x):
        # x: (batch, features)
        x = x.unsqueeze(1)                # (batch, seq=1, features)
        x = self.input_projection(x)      # (batch, 1, d_model)
        x = self.transformer(x)           # (batch, 1, d_model)
        _, h = self.gru(x)                # h: (1, batch, hidden)
        h = h.squeeze(0)                  # (batch, hidden)
        h = self.norm(h)
        out = self.fc(h)                  # RAW LOGITS
        return out
