import torch.nn as nn
from app.config import CONFIG

class CaptchaModel(nn.Module):
    def __init__(self, num_chars, dropout_rate=CONFIG.DROPOUT_RATE):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)
        )

        cnn_output_height = CONFIG.IMG_HEIGHT // 4
        self.reshape = nn.Linear(32 * cnn_output_height, 48)
        self.dropout = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(
            input_size=48,
            hidden_size=48,
            bidirectional=True,
            batch_first=True
        )

        self.classifier = nn.Linear(96, CONFIG.NUM_CLASSES + 1)  # +1 for blank token

    def forward(self, x):
        x = self.cnn(x)
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).reshape(batch_size, width, channels * height)
        x = self.reshape(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.classifier(x)
        return nn.functional.log_softmax(x, dim=2)
