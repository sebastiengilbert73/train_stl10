import torch
import einops

class ThreeConv2Lin(torch.nn.Module):
    def __init__(self, number_of_channels1=64, number_of_channels2=128, number_of_channels3=256,
                 linear1_size=1024, dropout_ratio=0.5):
        super(ThreeConv2Lin, self).__init__()
        self.number_of_channels1 = number_of_channels1
        self.number_of_channels2 = number_of_channels2
        self.number_of_channels3 = number_of_channels3
        self.linear1_size = linear1_size
        self.dropout_ratio = dropout_ratio

        self.conv1 = torch.nn.Conv2d(3, self.number_of_channels1, kernel_size=(3, 3), padding=1)
        self.conv2 = torch.nn.Conv2d(self.number_of_channels1, self.number_of_channels2, kernel_size=(3, 3), padding=1)
        self.conv3 = torch.nn.Conv2d(self.number_of_channels2, self.number_of_channels3, kernel_size=(3, 3), padding=1)
        self.batchnorm2d = torch.nn.BatchNorm2d(self.number_of_channels1)
        self.maxpool2d = torch.nn.MaxPool2d(2)
        self.linear1 = torch.nn.Linear(self.number_of_channels3 * 12 * 12, self.linear1_size)
        self.linear2 = torch.nn.Linear(self.linear1_size, 10)
        self.dropout1d = torch.nn.Dropout1d(p=dropout_ratio)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 3, 96, 96)
        act1 = self.conv1(input_tsr)  # (N, C1, 96, 96)
        act2 = self.maxpool2d(act1)  # (N, C1, 48, 48)
        act3 = torch.nn.functional.relu(act2)  # (N, C1, 48, 48)
        act4 = self.batchnorm2d(act3)  # (N, C1, 48, 48)

        act5 = self.conv2(act4)  # (N, C2, 48, 48)
        act6 = self.maxpool2d(act5)  # (N, C2, 24, 24)
        act7 = torch.nn.functional.relu(act6)  # (N, C2, 24, 24)

        act8 = self.conv3(act7)  # (N, C3, 24, 24)
        act9 = self.maxpool2d(act8)  # (N, C3, 12, 12)
        act10 = torch.nn.functional.relu(act9)  # (N, C3, 12, 12)

        act11 = einops.rearrange(act10, 'N C H W -> N (C H W)')  # (N, C3 * 12 * 12)
        act12 = self.linear1(act11)  # (N, linear1_size)
        act13 = torch.nn.functional.relu(act12)  # (N, linear1_size)
        act14 = self.dropout1d(act13)  # (N, linear1_size)

        act15 = self.linear2(act14)  # (N, 10)

        return act15
