"""
model.py
--------
Simple LSTM sequence classifier: takes a customer's recent order/return
history (SEQ_LEN steps x N_FEATURES) and outputs a single fraud-risk score
in [0, 1] for the return event being evaluated.
"""

import torch
import torch.nn as nn


class ReturnFraudLSTM(nn.Module):
    def __init__(self, n_features=11, hidden_size=32, num_layers=1, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        logits = self.head(last_hidden).squeeze(-1)  # (batch,)
        return logits  # raw logits, apply sigmoid outside
