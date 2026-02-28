import torch
import torch.nn as nn

from mini_katago.constants import BOARD_SIZE


class PolicyValueNetwork(nn.Module):
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
        self.action_size = board_size * board_size + 1  # + pass
        self.board_size = board_size

        hidden = 128
        self.trunk = nn.Sequential(
            # 10 convolution layers
            nn.Conv2d(
                in_channels=in_channels, out_channels=hidden, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden, out_channels=hidden, kernel_size=3, padding=1
            ),
            nn.BatchNorm2d(num_features=hidden),
            nn.ReLU(inplace=True),
        )

        # Policy head
        policy_out = 16
        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden, out_channels=policy_out, kernel_size=1),
            nn.BatchNorm2d(num_features=policy_out),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(
                in_features=policy_out * board_size * board_size,
                out_features=self.action_size,
            ),
        )

        # Value head
        value_out = 8
        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden, out_channels=value_out, kernel_size=1),
            nn.BatchNorm2d(num_features=value_out),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                in_features=value_out * board_size * board_size, out_features=hidden
            ),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=hidden, out_features=1),
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
        # x: (B, C, BOARD_SIZE, BOARD_SIZE)
        h = self.trunk(x)

        # Policy logits
        policy_logits = self.policy_head(h)  # (B, BOARD_SIZE * BOARD_SIZE + 1)

        # Value
        value = self.value_head(h).squeeze(1)  # (B,)

        return policy_logits, value
