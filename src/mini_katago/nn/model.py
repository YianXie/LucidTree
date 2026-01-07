from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallPVNet(nn.Module):
    def __init__(self, in_channels: int = 6, board_size: int = 9):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size + 1  # + pass

        hidden = 64
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Policy head
        self.policy_conv = nn.Conv2d(hidden, 2, 1)  # reduce channels
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)

        # Value head
        self.value_conv = nn.Conv2d(hidden, 1, 1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: Any) -> tuple[Any, torch.Tensor]:
        # x: (B, C, 9, 9)
        h = self.trunk(x)

        # Policy logits
        p = F.relu(self.policy_conv(h), inplace=True)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)  # (B, 82)

        # Value
        v = F.relu(self.value_conv(h), inplace=True)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v), inplace=True)
        value = torch.tanh(self.value_fc2(v)).squeeze(1)  # (B,)

        return policy_logits, value
