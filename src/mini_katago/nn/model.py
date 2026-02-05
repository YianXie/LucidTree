import torch
import torch.nn as nn

from mini_katago.constants import BOARD_SIZE


class SmallPVNet(nn.Module):
    """
    A small policy-value network for Go
    """

    def __init__(
        self,
        in_channels: int = 6,
        board_size: int = BOARD_SIZE,
    ) -> None:
        """
        Initialize the small policy-value network

        Args:
            in_channels (int, optional): the in_channels, or how many 'planes' of data is given. Defaults to 6.
            board_size (int, optional): the board size. Defaults to BOARD_SIZE.
        """
        super().__init__()
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
        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden, out_channels=2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(
                in_features=2 * board_size * board_size, out_features=self.action_size
            ),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden, out_channels=1, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Make one round of prediction

        Args:
            x (Any): the input data

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the policy logits, followed by the value
        """
        # x: (B, C, 9, 9)
        h = self.trunk(x)

        # Policy logits
        policy_logits = self.policy_head(h)  # (B, 82)

        # Value
        value = self.value_head(h).squeeze(1)  # (B,)

        return policy_logits, value
