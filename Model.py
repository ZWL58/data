import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn
import torch
import math
from SKAttention import SKAttention


# Four Conv
class AudioClassifier4_1(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        # nn.Conv2d(in_channles(几个频道，几层),out_channels（输出层数）,kernel_size（卷积层）,stride（步长）,padding)
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)) # 卷积层，其中2个输入通道、8个输出通道、5x5的卷积核、2个步长和2个填充的卷积层
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)   # BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Four Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=3)
        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)
        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)  # torch.Size([16, 64, 1, 1])
        x = x.view(x.shape[0], -1)    # torch.Size([16, 64])
        # Linear layer
        x = self.lin(x)    # torch.Size([16, 10])
        # Final output
        return x

# Five Conv
class AudioClassifier5_1(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        # nn.Conv2d(in_channles(几个频道，几层),out_channels（输出层数）,kernel_size（卷积层）,stride（步长）,padding)
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)) # 卷积层，其中2个输入通道、8个输出通道、5x5的卷积核、2个步长和2个填充的卷积层
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)   # BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Four Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Five Convolution Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        conv_layers += [self.conv5, self.relu5, self.bn5]


        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=128, out_features=3)
        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)
        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)  # torch.Size([16, 64, 1, 1])
        x = x.view(x.shape[0], -1)    # torch.Size([16, 64])
        # Linear layer
        x = self.lin(x)    # torch.Size([16, 10])
        # Final output
        return x

# Six Conv
class AudioClassifier6_1(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        # nn.Conv2d(in_channles(几个频道，几层),out_channels（输出层数）,kernel_size（卷积层）,stride（步长）,padding)
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)) # 卷积层，其中2个输入通道、8个输出通道、5x5的卷积核、2个步长和2个填充的卷积层
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)   # BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Four Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Five Convolution Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        conv_layers += [self.conv5, self.relu5, self.bn5]

        # Six Convolution Block
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(256)
        init.kaiming_normal_(self.conv6.weight, a=0.1)
        self.conv6.bias.data.zero_()
        conv_layers += [self.conv6, self.relu6, self.bn6]


        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=256, out_features=3)
        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)
        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)  # torch.Size([16, 64, 1, 1])
        x = x.view(x.shape[0], -1)    # torch.Size([16, 64])
        # Linear layer
        x = self.lin(x)    # torch.Size([16, 10])
        # Final output
        return x

# Seven Conv
class AudioClassifier7_1(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        # nn.Conv2d(in_channles(几个频道，几层),out_channels（输出层数）,kernel_size（卷积层）,stride（步长）,padding)
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)) # 卷积层，其中2个输入通道、8个输出通道、5x5的卷积核、2个步长和2个填充的卷积层
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)   # BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Four Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Five Convolution Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        conv_layers += [self.conv5, self.relu5, self.bn5]

        # Six Convolution Block
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(256)
        init.kaiming_normal_(self.conv6.weight, a=0.1)
        self.conv6.bias.data.zero_()
        conv_layers += [self.conv6, self.relu6, self.bn6]

        # Seven Convolution Block
        self.conv7 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu7 = nn.ReLU()
        self.bn7 = nn.BatchNorm2d(512)
        init.kaiming_normal_(self.conv6.weight, a=0.1)
        self.conv7.bias.data.zero_()
        conv_layers += [self.conv7, self.relu7, self.bn7]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=512, out_features=3)
        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)
        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)  # torch.Size([16, 64, 1, 1])
        x = x.view(x.shape[0], -1)    # torch.Size([16, 64])
        # Linear layer
        x = self.lin(x)    # torch.Size([16, 10])
        # Final output
        return x

# Conv(1-2)
class AudioClassifier12(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        # nn.Conv2d(in_channles(几个频道，几层),out_channels（输出层数）,kernel_size（卷积层）,stride（步长）,padding)
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)) # 卷积层，其中2个输入通道、8个输出通道、5x5的卷积核、2个步长和2个填充的卷积层
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)   # BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Four Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Five Convolution Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        conv_layers += [self.conv5, self.relu5, self.bn5]

        # Six Convolution Block
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(256)
        init.kaiming_normal_(self.conv6.weight, a=0.1)
        self.conv6.bias.data.zero_()
        conv_layers += [self.conv6, self.relu6, self.bn6]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=256, out_features=3)

        # Wrap the Convolutional Blocks
        # self.TripletAttention = TripletAttention()
        self.SKAttention = SKAttention(channel=8,reduction=16)
        # self.ShuffleAttention = ShuffleAttention(channel=8,G=2)
        self.conv0 = nn.Sequential(*conv_layers[:1])
        self.conv = nn.Sequential(*conv_layers[2:])

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv0(x)   # conv1后torch.Size([16, 8, 32, 172])，conv2后torch.Size([16, 16, 16, 86])，conv3后torch.Size([16, 32, 8, 43])，conv4后torch.Size([16, 64, 4, 22])
        # SKAttention注意力
        SKAttention = self.SKAttention
        x_SKAttention = SKAttention(x)
        # # TripletAttention注意力
        # x_TripletAttention = self.TripletAttention(x)
        # # ShuffleAttention注意力
        # ShuffleAttention = self.ShuffleAttention
        # x_ShuffleAttention = ShuffleAttention(x)

        x = self.conv(x)
        x_SKAttention = self.conv(x_SKAttention)
        # x_TripletAttention = self.conv(x_TripletAttention)
        # x_ShuffleAttention= self.conv(x_ShuffleAttention)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)  # torch.Size([16, 64, 1, 1])
        x_SKAttention = self.ap(x_SKAttention)
        # x_TripletAttention = self.ap(x_TripletAttention)
        # x_ShuffleAttention = self.ap(x_ShuffleAttention)

        x = x.view(x.shape[0], -1)    # torch.Size([16, 64])
        x_SKAttention = x_SKAttention.view(x.shape[0], -1)
        # x_TripletAttention = x_TripletAttention.view(x.shape[0], -1)
        # x_ShuffleAttention = x_ShuffleAttention.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)    # torch.Size([16, 10])
        x_SKAttention = self.lin(x_SKAttention)
        # x_TripletAttention = self.lin(x_TripletAttention)
        # x_ShuffleAttention = self.lin(x_ShuffleAttention)

        # Final output
        return x_SKAttention    # x_TripletAttention,x_ShuffleAttention

# Conv(2-3)
class AudioClassifier23(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []
        conv_layers1 = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        # nn.Conv2d(in_channles(几个频道，几层),out_channels（输出层数）,kernel_size（卷积层）,stride（步长）,padding)
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)) # 卷积层，其中2个输入通道、8个输出通道、5x5的卷积核、2个步长和2个填充的卷积层
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)   # BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers1 += [self.conv3, self.relu3, self.bn3]

        # Four Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers1 += [self.conv4, self.relu4, self.bn4]

        # Five Convolution Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        conv_layers1 += [self.conv5, self.relu5, self.bn5]

        # Six Convolution Block
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(256)
        init.kaiming_normal_(self.conv6.weight, a=0.1)
        self.conv6.bias.data.zero_()
        conv_layers1 += [self.conv6, self.relu6, self.bn6]


        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        # self.dropout = nn.Dropout(p=0.5)  # dropout训练
        self.lin = nn.Linear(in_features=256, out_features=3)

        # Wrap the Convolutional Blocks
        # self.TripletAttention = TripletAttention()
        self.SKAttention = SKAttention(channel=16,reduction=16)
        # self.SEAttention = SEAttention(channel=16, reduction=8)
        # self.ShuffleAttention = ShuffleAttention(channel=16,G=2)
        self.conv0 = nn.Sequential(*conv_layers)
        self.conv1 = nn.Sequential(*conv_layers1)


    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x1):
        # Run the convolutional blocks
        x1 = self.conv0(x1)   # conv1后torch.Size([16, 8, 32, 172])，conv2后torch.Size([16, 16, 16, 86])，conv3后torch.Size([16, 32, 8, 43])，conv4后torch.Size([16, 64, 4, 22])
        # SKAttention注意力
        SKAttention = self.SKAttention
        x1_SKAttention = SKAttention(x1)
        # # TripletAttention注意力
        # x1_TripletAttention = self.TripletAttention(x1)
        # # ShuffleAttention注意力
        # ShuffleAttention = self.ShuffleAttention
        # x1_ShuffleAttention = ShuffleAttention(x1)
        # # SEAttention注意力
        # SEAttention = self.SEAttention
        # x1_SEAttention = SEAttention(x1)

        x1 = self.conv1(x1)
        x1_SKAttention = self.conv1(x1_SKAttention)
        # x1_SEAttention = self.conv1(x1_SEAttention)
        # x1_TripletAttention = self.conv1(x1_TripletAttention)
        # x1_ShuffleAttention= self.conv1(x1_ShuffleAttention)

        # Adaptive pool and flatten for input to linear layer
        x1 = self.ap(x1)  # torch.Size([16, 64, 1, 1])
        x_SKAttention = self.ap(x1_SKAttention)
        # x_SEAttention = self.ap(x1_SEAttention)
        # x_TripletAttention = self.ap(x1_TripletAttention)
        # x_ShuffleAttention = self.ap(x1_ShuffleAttention)

        x1 = x1.view(x1.shape[0], -1)    # torch.Size([16, 64])
        x1_SKAttention = x_SKAttention.view(x1.shape[0], -1)
        # x1_SEAttention = x_SEAttention.view(x1.shape[0], -1)
        # x1_TripletAttention = x_TripletAttention.view(x1.shape[0], -1)
        # x1_ShuffleAttention = x_ShuffleAttention.view(x1.shape[0], -1)

        # Linear layer
        x1 = self.lin(x1)    # torch.Size([16, 10])
        x1_SKAttention = self.lin(x1_SKAttention)
        # x1_SEAttention = self.lin(x1_SEAttention)
        # x1_TripletAttention = self.lin(x1_TripletAttention)
        # x1_ShuffleAttention = self.lin(x1_ShuffleAttention)

        # x1_SKAttention = self.dropout(x1_SKAttention)

        # Final output
        return x1_SKAttention     # x1, x1_TripletAttention,x1_ShuffleAttention

# Conv(3-4)
class AudioClassifier34(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []
        conv_layers1 = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        # nn.Conv2d(in_channles(几个频道，几层),out_channels（输出层数）,kernel_size（卷积层）,stride（步长）,padding)
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)) # 卷积层，其中2个输入通道、8个输出通道、5x5的卷积核、2个步长和2个填充的卷积层
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)   # BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Four Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers1 += [self.conv4, self.relu4, self.bn4]

        # Five Convolution Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        conv_layers1 += [self.conv5, self.relu5, self.bn5]

        # Six Convolution Block
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(256)
        init.kaiming_normal_(self.conv6.weight, a=0.1)
        self.conv6.bias.data.zero_()
        conv_layers1 += [self.conv6, self.relu6, self.bn6]


        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=256, out_features=3)

        # Wrap the Convolutional Blocks
        # self.TripletAttention = TripletAttention()
        self.SKAttention = SKAttention(channel=32,reduction=16)
        # self.ShuffleAttention = ShuffleAttention(channel=32,G=2)
        self.conv0 = nn.Sequential(*conv_layers)
        self.conv1 = nn.Sequential(*conv_layers1)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv0(x)   # conv1后torch.Size([16, 8, 32, 172])，conv2后torch.Size([16, 16, 16, 86])，conv3后torch.Size([16, 32, 8, 43])，conv4后torch.Size([16, 64, 4, 22])
        # SKAttention注意力
        SKAttention = self.SKAttention
        x_SKAttention = SKAttention(x)
        # # TripletAttention注意力
        # x_TripletAttention = self.TripletAttention(x)
        # # ShuffleAttention注意力
        # ShuffleAttention = self.ShuffleAttention
        # x_ShuffleAttention = ShuffleAttention(x)

        x = self.conv1(x)
        x_SKAttention = self.conv1(x_SKAttention)
        # x_TripletAttention = self.conv1(x_TripletAttention)
        # x_ShuffleAttention= self.conv1(x_ShuffleAttention)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)  # torch.Size([16, 64, 1, 1])
        x_SKAttention = self.ap(x_SKAttention)
        # x_TripletAttention = self.ap(x_TripletAttention)
        # x_ShuffleAttention = self.ap(x_ShuffleAttention)

        x = x.view(x.shape[0], -1)    # torch.Size([16, 64])
        x_SKAttention = x_SKAttention.view(x.shape[0], -1)
        # x_TripletAttention = x_TripletAttention.view(x.shape[0], -1)
        # x_ShuffleAttention = x_ShuffleAttention.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)    # torch.Size([16, 10])
        x_SKAttention = self.lin(x_SKAttention)
        # x_TripletAttention = self.lin(x_TripletAttention)
        # x_ShuffleAttention = self.lin(x_ShuffleAttention)

        # Final output
        return x_SKAttention    # x,x_TripletAttention,x_ShuffleAttention

# Conv(4-5)
class AudioClassifier45(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []
        conv_layers1 = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        # nn.Conv2d(in_channles(几个频道，几层),out_channels（输出层数）,kernel_size（卷积层）,stride（步长）,padding)
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)) # 卷积层，其中2个输入通道、8个输出通道、5x5的卷积核、2个步长和2个填充的卷积层
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)   # BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Four Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Five Convolution Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        conv_layers1 += [self.conv5, self.relu5, self.bn5]

        # Six Convolution Block
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(256)
        init.kaiming_normal_(self.conv6.weight, a=0.1)
        self.conv6.bias.data.zero_()
        conv_layers1 += [self.conv6, self.relu6, self.bn6]


        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=256, out_features=3)

        # Wrap the Convolutional Blocks
        # self.TripletAttention = TripletAttention()
        self.SKAttention = SKAttention(channel=64,reduction=16)
        # self.ShuffleAttention = ShuffleAttention(channel=64,G=2)
        # self.conv = nn.Sequential(*conv_layers[:1])
        self.conv0 = nn.Sequential(*conv_layers)
        self.conv1 = nn.Sequential(*conv_layers1)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv0(x)   # conv1后torch.Size([16, 8, 32, 172])，conv2后torch.Size([16, 16, 16, 86])，conv3后torch.Size([16, 32, 8, 43])，conv4后torch.Size([16, 64, 4, 22])

        # SKAttention注意力
        SKAttention = self.SKAttention
        x_SKAttention = SKAttention(x)
        # # TripletAttention注意力
        # x_TripletAttention = self.TripletAttention(x)
        # # ShuffleAttention注意力
        # ShuffleAttention = self.ShuffleAttention
        # x_ShuffleAttention = ShuffleAttention(x)

        x = self.conv1(x)

        x_SKAttention = self.conv1(x_SKAttention)
        # x_TripletAttention = self.conv1(x_TripletAttention)
        # x_ShuffleAttention= self.conv1(x_ShuffleAttention)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)  # torch.Size([16, 64, 1, 1])
        x_SKAttention = self.ap(x_SKAttention)
        # x_TripletAttention = self.ap(x_TripletAttention)
        # x_ShuffleAttention = self.ap(x_ShuffleAttention)

        x = x.view(x.shape[0], -1)    # torch.Size([16, 64])
        x_SKAttention = x_SKAttention.view(x.shape[0], -1)
        # x_TripletAttention = x_TripletAttention.view(x.shape[0], -1)
        # x_ShuffleAttention = x_ShuffleAttention.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)    # torch.Size([16, 10])
        x_SKAttention = self.lin(x_SKAttention)
        # x_TripletAttention = self.lin(x_TripletAttention)
        # x_ShuffleAttention = self.lin(x_ShuffleAttention)

        # Final output
        return x_SKAttention    # x,x_TripletAttention,x_ShuffleAttention

# Conv(5-6)
class AudioClassifier56(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []
        conv_layers1 = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        # nn.Conv2d(in_channles(几个频道，几层),out_channels（输出层数）,kernel_size（卷积层）,stride（步长）,padding)
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)) # 卷积层，其中2个输入通道、8个输出通道、5x5的卷积核、2个步长和2个填充的卷积层
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)   # BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Four Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Five Convolution Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        conv_layers += [self.conv5, self.relu5, self.bn5]

        # Six Convolution Block
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(256)
        init.kaiming_normal_(self.conv6.weight, a=0.1)
        self.conv6.bias.data.zero_()
        conv_layers1 += [self.conv6, self.relu6, self.bn6]


        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=256, out_features=3)

        # Wrap the Convolutional Blocks
        # self.TripletAttention = TripletAttention()
        self.SKAttention = SKAttention(channel=128,reduction=16)
        # self.ShuffleAttention = ShuffleAttention(channel=128,G=2)
        # self.conv = nn.Sequential(*conv_layers[:1])
        self.conv0 = nn.Sequential(*conv_layers)
        self.conv1 = nn.Sequential(*conv_layers1)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv0(x)   # conv1后torch.Size([16, 8, 32, 172])，conv2后torch.Size([16, 16, 16, 86])，conv3后torch.Size([16, 32, 8, 43])，conv4后torch.Size([16, 64, 4, 22])

        # SKAttention注意力
        SKAttention = self.SKAttention
        x_SKAttention = SKAttention(x)
        # # TripletAttention注意力
        # x_TripletAttention = self.TripletAttention(x)
        # # ShuffleAttention注意力
        # ShuffleAttention = self.ShuffleAttention
        # x_ShuffleAttention = ShuffleAttention(x)

        x = self.conv1(x)

        x_SKAttention = self.conv1(x_SKAttention)
        # x_TripletAttention = self.conv1(x_TripletAttention)
        # x_ShuffleAttention= self.conv1(x_ShuffleAttention)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)  # torch.Size([16, 64, 1, 1])
        x_SKAttention = self.ap(x_SKAttention)
        # x_TripletAttention = self.ap(x_TripletAttention)
        # x_ShuffleAttention = self.ap(x_ShuffleAttention)

        x = x.view(x.shape[0], -1)    # torch.Size([16, 64])
        x_SKAttention = x_SKAttention.view(x.shape[0], -1)
        # x_TripletAttention = x_TripletAttention.view(x.shape[0], -1)
        # x_ShuffleAttention = x_ShuffleAttention.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)    # torch.Size([16, 10])
        x_SKAttention = self.lin(x_SKAttention)
        # x_TripletAttention = self.lin(x_TripletAttention)
        # x_ShuffleAttention = self.lin(x_ShuffleAttention)

        # Final output
        return x_SKAttention    # x,x_TripletAttention,x_ShuffleAttention

# No Attention
class AudioClassifier(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        # nn.Conv2d(in_channles(几个频道，几层),out_channels（输出层数）,kernel_size（卷积层）,stride（步长）,padding)
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)) # 卷积层，其中2个输入通道、8个输出通道、5x5的卷积核、2个步长和2个填充的卷积层
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)   # BatchNorm2d进行数据的归一化处理，这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Four Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Five Convolution Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        conv_layers += [self.conv5, self.relu5, self.bn5]

        # Six Convolution Block
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.pooling4 = nn.MaxPool2d(2)
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(256)
        init.kaiming_normal_(self.conv6.weight, a=0.1)
        self.conv6.bias.data.zero_()
        conv_layers += [self.conv6, self.relu6, self.bn6]


        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=256, out_features=3)
        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)
        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)  # torch.Size([16, 64, 1, 1])
        x = x.view(x.shape[0], -1)    # torch.Size([16, 64])
        # Linear layer
        x = self.lin(x)    # torch.Size([16, 10])
        # Final output
        return x