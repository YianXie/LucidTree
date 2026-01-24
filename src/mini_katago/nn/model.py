import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_katago.constants import BOARD_SIZE


class SmallPVNet(nn.Module):
    def __init__(
        self, in_channels: int = 6, board_size: int = BOARD_SIZE, *, seed: int = 0
    ) -> None:
        """
        Initialize the small policy-value network

        Args:
            in_channels (int, optional): the in_channels, or how many 'planes' of data is given. Defaults to 6.
            board_size (int, optional): the board size. Defaults to BOARD_SIZE.
            seed (int, optional): the seed for PyTorch. Defaults to 0.
        """
        super().__init__()
        torch.manual_seed(seed)
        self.board_size = board_size
        self.action_size = board_size * board_size + 1  # + pass

        hidden = 64
        self.trunk = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=hidden, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
        )

        # Policy head
        self.policy_conv = nn.Conv2d(in_channels=hidden, out_channels=2, kernel_size=1)
        self.policy_fc = nn.Linear(
            in_features=2 * board_size * board_size, out_features=self.action_size
        )

        # Value head
        self.value_conv = nn.Conv2d(hidden, 1, 1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Make one round of prediction

        Args:
            x (Any): the input data

        Returns:
            tuple[Any, torch.Tensor]: the policy logits, followed by the value
        """
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
