from torch import nn
import torch.nn.functional as F

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, args.num_classes)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, args.num_classes)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# class modelC(nn.Module):
#     def __init__(self, input_size, **kwargs):
#         super(modelC, self).__init__()
#         self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
#         self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
#         self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)
#         self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
#         self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
#         self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)
#         self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
#         self.conv8 = nn.Conv2d(192, 192, 1)

#         self.class_conv = nn.Conv2d(192, args.num_classes, 1)


#     def forward(self, x):
#         x_drop = F.dropout(x, .2)
#         conv1_out = F.relu(self.conv1(x_drop))
#         conv2_out = F.relu(self.conv2(conv1_out))
#         conv3_out = F.relu(self.conv3(conv2_out))
#         conv3_out_drop = F.dropout(conv3_out, .5)
#         conv4_out = F.relu(self.conv4(conv3_out_drop))
#         conv5_out = F.relu(self.conv5(conv4_out))
#         conv6_out = F.relu(self.conv6(conv5_out))
#         conv6_out_drop = F.dropout(conv6_out, .5)
#         conv7_out = F.relu(self.conv7(conv6_out_drop))
#         conv8_out = F.relu(self.conv8(conv7_out))

#         class_out = F.relu(self.class_conv(conv8_out))
#         pool_out = F.adaptive_avg_pool2d(class_out, 1)
#         pool_out.squeeze_(-1)
#         pool_out.squeeze_(-1)
#         return pool_out
