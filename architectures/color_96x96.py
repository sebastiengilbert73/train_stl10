import torch
import einops
from collections import OrderedDict

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

class ResidualBlock3(torch.nn.Module):
    def __init__(self, number_of_channels=64):
        super(ResidualBlock3, self).__init__()
        self.conv1 = torch.nn.Conv2d(number_of_channels, number_of_channels, kernel_size=(3, 3), padding=1)
        self.batchnorm1 = torch.nn.BatchNorm2d(number_of_channels)
        self.conv2 = torch.nn.Conv2d(number_of_channels, number_of_channels, kernel_size=(3, 3), padding=1)
        self.batchnorm2 = torch.nn.BatchNorm2d(number_of_channels)

    def forward(self, input_tsr):  # input_tsr.shape = (N, C, H, W)
        act1 = self.conv1(input_tsr)  # (N, C, H, W)
        act2 = self.batchnorm1(act1)  # (N, C, H, W)
        act3 = torch.nn.functional.relu(act2)  # (N, C, H, W)

        act4 = self.conv2(act3)  # (N, C, H, W)
        act5 = self.batchnorm2(act4)  # (N, C, H, W)
        act6 = torch.nn.functional.relu(act5)  # (N, C, H, W)
        act7 = act6 + input_tsr  # (N, C, H, W)
        return act7

class ThreeRes1Lin(torch.nn.Module):
    def __init__(self, residual_channels=64):
        super(ThreeRes1Lin, self).__init__()
        self.residual_channels = residual_channels
        self.conv1 = torch.nn.Conv2d(3, residual_channels, kernel_size=(3, 3), padding=1)
        self.residual1 = ResidualBlock3(self.residual_channels)
        self.residual2 = ResidualBlock3(self.residual_channels)
        self.residual3 = ResidualBlock3(self.residual_channels)
        self.adaptiveavgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear1 = torch.nn.Linear(self.residual_channels, 10)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 3, H, W)
        act1 = self.conv1(input_tsr)  # (N, C, H, W)
        act2 = torch.nn.functional.relu(act1)  # (N, C, H, W)
        act3 = self.residual1(act2)  # (N, C, H, W)
        act4 = self.residual2(act3)  # (N, C, H, W)
        act5 = self.residual3(act4)  # (N, C, H, W)
        act6 = self.adaptiveavgpool(act5)  # (N, C, 1, 1)
        act7 = einops.rearrange(act6, 'N C H W -> N (C H W)', H=1, W=1)  # (N, C)
        act8 = self.linear1(act7)  # (N, 10)
        return act8

class Resx6Linx1(torch.nn.Module):
    def __init__(self, residual_channels=64):
        super(Resx6Linx1, self).__init__()
        self.residual_channels = residual_channels
        self.conv1 = torch.nn.Conv2d(3, residual_channels, kernel_size=(3, 3), padding=1)
        self.residual1 = ResidualBlock3(self.residual_channels)
        self.residual2 = ResidualBlock3(self.residual_channels)
        self.residual3 = ResidualBlock3(self.residual_channels)
        self.residual4 = ResidualBlock3(self.residual_channels)
        self.residual5 = ResidualBlock3(self.residual_channels)
        self.residual6 = ResidualBlock3(self.residual_channels)
        self.adaptiveavgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear1 = torch.nn.Linear(self.residual_channels, 10)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 3, H, W)
        act1 = self.conv1(input_tsr)  # (N, C, H, W)
        act2 = torch.nn.functional.relu(act1)  # (N, C, H, W)
        act3 = self.residual1(act2)  # (N, C, H, W)
        act4 = self.residual2(act3)  # (N, C, H, W)
        act5 = self.residual3(act4)  # (N, C, H, W)
        act6 = self.residual4(act5)  # (N, C, H, W)
        act7 = self.residual5(act6)  # (N, C, H, W)
        act8 = self.residual6(act7)  # (N, C, H, W)
        act9 = self.adaptiveavgpool(act8)  # (N, C, 1, 1)
        act10 = einops.rearrange(act9, 'N C H W -> N (C H W)', H=1, W=1)  # (N, C)
        act11 = self.linear1(act10)  # (N, 10)
        return act11

class Resx3x3Linx1(torch.nn.Module):
    def __init__(self, residual_channels123=64, residual_channels456=32):
        super(Resx3x3Linx1, self).__init__()
        self.residual_channels123 = residual_channels123
        self.residual_channels456 = residual_channels456
        self.conv1 = torch.nn.Conv2d(3, self.residual_channels123, kernel_size=(3, 3), padding=1)
        self.residual1 = ResidualBlock3(self.residual_channels123)
        self.residual2 = ResidualBlock3(self.residual_channels123)
        self.residual3 = ResidualBlock3(self.residual_channels123)
        self.conv2 = torch.nn.Conv2d(self.residual_channels123, self.residual_channels456,
                                       kernel_size=(3, 3), padding=1)
        self.residual4 = ResidualBlock3(self.residual_channels456)
        self.residual5 = ResidualBlock3(self.residual_channels456)
        self.residual6 = ResidualBlock3(self.residual_channels456)
        self.adaptiveavgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.linear1 = torch.nn.Linear(self.residual_channels456, 10)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 3, H, W)
        act1 = self.conv1(input_tsr)  # (N, C123, H, W)
        act2 = torch.nn.functional.relu(act1)  # (N, C, H, W)
        act3 = self.residual1(act2)  # (N, C123, H, W)
        act4 = self.residual2(act3)  # (N, C123, H, W)
        act5 = self.residual3(act4)  # (N, C123, H, W)
        act6 = self.conv2(act5)  # (N, C456, H, W)
        act7 = torch.nn.functional.relu(act6)  # (N, C456, H, W)
        act8 = self.residual4(act7)  # (N, C456, H, W)
        act9 = self.residual5(act8)  # (N, C456, H, W)
        act10 = self.residual6(act9)  # (N, C456, H, W)
        act11 = self.adaptiveavgpool(act10)  # (N, C456, 1, 1)
        act12 = einops.rearrange(act11, 'N C H W -> N (C H W)', H=1, W=1)  # (N, C456)
        act13 = self.linear1(act12)  # (N, 10)
        return act13

class ResTrios(torch.nn.Module):
    def __init__(self, residual_channels_list=[64, 32, 16]):
        super(ResTrios, self).__init__()
        if len(residual_channels_list) == 0:
            raise ValueError(f"len(residual_channels_list) == 0")
        self.residual_channels_list = residual_channels_list
        self.residual_block_trios = torch.nn.ParameterList()
        self.adaptiveavgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(self.residual_channels_list[-1], 10)

        for trio_ndx in range(len(self.residual_channels_list)):
            number_of_inputs = 3
            if trio_ndx > 0:
                number_of_inputs = self.residual_channels_list[trio_ndx - 1]
            number_of_channels = self.residual_channels_list[trio_ndx]
            trio_sequence = torch.nn.Sequential(
                OrderedDict([
                    ('conv', torch.nn.Conv2d(number_of_inputs, number_of_channels, kernel_size=(3, 3), padding=1)),
                    ('relu', torch.nn.ReLU()),
                    ('res1', ResidualBlock3(number_of_channels)),
                    ('res2', ResidualBlock3(number_of_channels)),
                    ('res3', ResidualBlock3(number_of_channels))
                ])
            )
            self.residual_block_trios.append(trio_sequence)

    def forward(self, input_tsr):  # input_tsr.shape = (N, 3, H, W)
        act = input_tsr
        for trio in self.residual_block_trios:
            act = trio(act)
        # act.shape = (N, C_last, H, W)
        act1 = self.adaptiveavgpool(act)  # (N, C_last, 1, 1)
        act2 = einops.rearrange(act1, 'N C H W -> N (C H W)', H=1, W=1)  # (N, C_last)
        act3 = self.fc(act2)  # (N, 10)
        return act3