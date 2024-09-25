import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class simpleConvolutional(nn.Module):
    def __init__(self,input_size, output_size, normalize_features=False):
        super(simpleConvolutional, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=5, stride = 1,padding=3)
        self.bn1 = nn.BatchNorm1d(input_size)
        self.conv2 = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=5, stride = 1,padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride = 1,padding=3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride = 1, padding=3)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 2, 128)
        self.fc2 = nn.Linear(128, 64)  # input_size here corresponds to the number of features
        self.fc3 = nn.Linear(64, output_size)
        self.normalize_features = normalize_features
        self.relu = nn.ReLU()
        self.normalize_features = normalize_features
        self.drop = nn.Dropout(0.25)
        
    def forward(self, x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.drop(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.drop(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.pool(x)
        x = self.drop(x)
        x = x.view(-1, 512 ) 
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        x = x.squeeze()
        return x
    


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, normalize_features=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.normalize_features = normalize_features

    def forward(self, x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x = x.permute(0, 2, 1)  # Reshape input for LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.drop(out)
        out = self.fc1(out)
        out = out.squeeze()
        return out

class BidirectionalLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=True, normalize_features=False):
        super(BidirectionalLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=0.25)
        self.drop = nn.Dropout(0.25)  # Increased dropout
        # Adjusting the input feature size of the fully connected layer if LSTM is bidirectional
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.normalize_features = normalize_features
        self.relu = nn.ReLU(0.1)

    def forward(self, x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True)
        x = x.permute(0, 2, 1)  # Reshape input for LSTM layer
        out, _ = self.lstm(x)
        # out = self.drop(out)
        # Selecting the last output from sequence, considering bidirectional case
        out = out[:, -1, :]
        out = self.drop(out)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = out.squeeze()
        return out

class DepthwiseConvLSTM(nn.Module):
    def __init__(self, num_features, seq_length, hidden_size, num_layers, output_size, normalize_features=False):
        super(DepthwiseConvLSTM, self).__init__()
        self.depthwise_conv = nn.Conv1d(num_features, 2 * num_features,  kernel_size=5, groups=num_features, padding='same')
        self.pointwise_conv = nn.Conv1d(2 * num_features, 2 * num_features, kernel_size=1)
        self.bn = nn.BatchNorm1d(2 * num_features)
        self.lstm = nn.LSTM(2 * num_features, hidden_size, num_layers, batch_first=True, dropout=0.25)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.normalize_features = normalize_features
        self.drop = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, 1, num_features, seq_length)
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.drop(x)
        # Flatten the features and datapoints dimensions
        x = x.permute(0,2,1)
        # LSTM expects input of shape (batch, seq_len, features)
        x, _ = self.lstm(x)
        # Take the output of the last LSTM cell
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = x.squeeze()
        return x

class CNNLSTMModel(nn.Module):
    def __init__(self, input_channels, seq_length, hidden_size, num_layers, output_size, normalize_features=False):
        super(CNNLSTMModel, self).__init__()
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.normalize_features = normalize_features
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, padding='same')
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        # self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        # self.normalize_features = normalize_features
        
        # LSTM layer
        self.lstm_input_size = 512  
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.25)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layer
        self.fc1 = nn.Linear(hidden_size, 128)
        
        self.fc2 = nn.Linear(128, output_size)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True)
        
        # CNN layers
        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        # x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        # x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.bn3(x)
        # x = self.pool(x)
        
        # Reshape output for LSTM layer        
        x = x.permute(0, 2, 1)
        
        # LSTM layer
        x, _ = self.lstm(x)
        
        # Dropout and fully connected layer
        x = x[:, -1, :]  # Take the output of the last time step
        x = self.dropout(x)
        x = self.relu(self.fc1(x))  # Take the output of the last time step
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.squeeze()
        return x



    
class ComplexCNNLSTMModel(nn.Module):
    def __init__(self,input_channels, seq_length, hidden_size, num_layers, output_size, normalize_features= True):
        super(ComplexCNNLSTMModel, self).__init__()
        
        # inputs
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.normalize_features = normalize_features
        
        # Define CNN layers
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)  # Batch normalization
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)  # Batch normalization
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(256)  # Batch normalization
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout_cnn = nn.Dropout(0.25)  # Dropout for CNN layers
                        
        # LSTM layer
        self.lstm_input_size = 2048 
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.25)  # Dropout for LSTM layer
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout_fc = nn.Dropout(0.5)  # Dropout for fully connected layers
        self.fc2 = nn.Linear(64, output_size)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True) 
        x = x.view(-1, 1, x.size(1), x.size(2)) 
        
        # Apply CNN layers with skip connection
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x3 = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x3)) + x3))  # Skip connection
        x = self.dropout_cnn(x)
        
        # Reshape for LSTM layer
        # x will be of shape (batch_size, 256 * 8, 5)
        # x = x.view(x.size(0), -1, 256 * 8 * 5)
        x = x.view(x.size(0), -1,x.size(3) )
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout_lstm(x)  # Apply dropout after selecting the last time step
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)  # Apply dropout before the final layer
        x = self.fc2(x)
        x = x.squeeze()
        return x
 
class ComplexCNNBiDirLSTMModel(nn.Module):
    def __init__(self,input_channels, seq_length, hidden_size, num_layers, output_size, normalize_features= True):
        super(ComplexCNNBiDirLSTMModel, self).__init__()
        
        # inputs
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.normalize_features = normalize_features
        
        # Define CNN layers
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(128)  # Batch normalization
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(256)  # Batch normalization
        self.conv4 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(256)  # Batch normalization
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout_cnn = nn.Dropout(0.25)  # Dropout for CNN layers
                        
        # LSTM layer
        self.lstm_input_size = 2048 
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.25)
        self.dropout_lstm = nn.Dropout(0.25)  # Dropout for LSTM layer
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size*2, 64)
        self.dropout_fc = nn.Dropout(0.5)  # Dropout for fully connected layers
        self.fc2 = nn.Linear(64, output_size)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True) 
        x = x.view(-1, 1, x.size(1), x.size(2)) 
        
        # Apply CNN layers with skip connection
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x3 = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x3)) + x3))  # Skip connection
        x = self.dropout_cnn(x)
        
        # Reshape for LSTM layer
        # x = x.view(x.size(0), -1, 256 * 8 * 5)
        x = x.view(x.size(0), -1,x.size(3) )
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout_lstm(x)  # Apply dropout after selecting the last time step
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)  # Apply dropout before the final layer
        x = self.fc2(x)
        x = x.squeeze()
        return x

class CNNGRUModel_bidirectional(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, bidirectional=True, normalize_features=False):
        super(CNNGRUModel_bidirectional, self).__init__()
        
        # First CNN layer
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=1, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second CNN layer
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, padding='same')
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropCNN = nn.Dropout(0.25)
        
        # Bidirectional GRU layer
        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)
                
        # Fully Connected layer
        self.fc1 = nn.Linear(256 * 2, 256)  # 50 * 2 because the GRU is bidirectional
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.normalize_features = normalize_features
        self.drop = nn.Dropout(0.25)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True)
        # Pass input through the first CNN layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.drop(x)
        
        # Pass through the second CNN layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.dropCNN(x)
        
        # Reshape for GRU layer
        x = x.permute(0, 2, 1)  # Reshape input for GRU layer
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.drop(x)
        x = self.relu(x)  # Apply ReLU after the first FC layer
        x = self.fc2(x)
        x = self.relu(x)  # Apply ReLU after the second FC layer
        x = self.fc3(x)
        x = x.squeeze()
        
        return x

class OneDimCNNModel(nn.Module):
    def __init__(self, input_channels, output_size, normalize_features=False):
        super(OneDimCNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=1, stride = 1, padding='same')
        self.bn1 = nn.BatchNorm1d(64)  # Batch normalization
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, padding='same')
        self.bn2 = nn.BatchNorm1d(128)  # Batch normalization
        self.normalize_features = normalize_features
        self.fc1 = nn.Linear(128 * 10, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.AdaptiveAvgPool1d(10)

    def forward(self, x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True)
        # x = x.view(x.size(0), 1, x.size(1), x.size(2)) 
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(x)
        x = x.view(-1, 128 * 10)  # Flatten the tensor for the fully connected layer
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze()
        return x

# Define the ResNet block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResNetBlock, self).__init__()
        self.conv0 = nn.Conv1d(in_channels, out_channels, kernel_size=21, padding=10)
        self.bn0 = nn.BatchNorm1d(out_channels)
        self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.downsample = downsample
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn0(self.conv0(x)))
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            identity = self.shortcut(x)
        out += identity
        out = self.relu(out)
        return out


    
# Define the full ResNet model
class ResNet(nn.Module):
    def __init__(self, input_shape, nb_classes, normalize_features=False):
        super(ResNet, self).__init__()
        self.block1 = ResNetBlock(input_shape[0], 32, downsample=True)
        self.block2 = ResNetBlock(32, 64, downsample=True)
        self.block3 = ResNetBlock(64, 128, downsample=True)
        self.drop = nn.Dropout(0.25)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, nb_classes)
        self.normalize_features = normalize_features
        self.relu = nn.ReLU()
        
    def forward(self, x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True)
        x = self.block1(x)
        x = self.block2(x)
        x = self.drop(x)
        x = self.block3(x)
        x = self.drop(x)
        x = self.gap(x).squeeze(-1)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.squeeze()
        return x
class ResNet_LSTM(nn.Module):
    def __init__(self, input_shape, nb_classes, normalize_features=False):
        super(ResNet_LSTM, self).__init__()
        self.block1 = ResNetBlock(input_shape[0], 64, downsample=True)
        self.block2 = ResNetBlock(64, 128, downsample=True)
        self.block3 = ResNetBlock(128, 256, downsample=True)
        self.drop = nn.Dropout(0.25)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.normalize_features = normalize_features
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True, dropout=0.25, bidirectional=True)
        self.fc1 = nn.Linear(256*2, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, nb_classes)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True)
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.drop(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.drop(x)
        # x = self.gap(x).squeeze(-1)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.drop(x)
        x = self.leakyrelu(self.fc1(x))
        x = self.drop(x)
        x = self.leakyrelu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        x = x.squeeze()
        return x    
    
# Define the ResNet block
class ResNetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResNetBlock2D, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,21), padding='same')
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,9), padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,5), padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding='same')
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1,1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn0(self.conv0(x)))
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample:
            identity = self.shortcut(x)
        out += identity
        out = self.relu(out)
        return out
    
# Define the full ResNet model
class ResNet2D(nn.Module):
    def __init__(self, input_shape, nb_classes, normalize_features=False):
        super(ResNet2D, self).__init__()
        self.block1 = ResNetBlock2D(1, 64, downsample=True)
        self.block2 = ResNetBlock2D(64, 64, downsample=True)
        self.block3 = ResNetBlock2D(64, 128, downsample=True)
        self.block4 = ResNetBlock2D(128, 128, downsample=True)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, nb_classes)
        self.normalize_features = normalize_features
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.gap = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True)
        x = x.view(-1, 1, x.size(1), x.size(2))
        x = self.block1(x)
        x = self.pool(x)
        x = self.drop(x)
        x = self.block2(x)
        x = self.pool(x)
        x = self.drop(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.drop(x)
        x = self.block4(x)
        x = self.pool(x)
        x = self.drop(x)
        x = self.gap(x)
        x = x.view(x.size(0),-1)
        x = self.drop(x)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = x.squeeze()
        return x
    


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
                
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResidualCNNLSTMModel(nn.Module):
    def __init__(self, input_channels, seq_length, hidden_size, num_layers, output_size, normalize_features=True):
        super(ResidualCNNLSTMModel, self).__init__()
        
        self.input_channels = input_channels
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.normalize_features = normalize_features
        
        self.residual_block1 = ResidualBlock(self.input_channels, 64)
        self.residual_block2 = ResidualBlock(64, 128)
        self.residual_block3 = ResidualBlock(128, 256)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.dropout_cnn = nn.Dropout(0.25)
        
        self.lstm = nn.LSTM(input_size=1024, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout_lstm = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(hidden_size*2, 128)
        self.dropout_fc = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True)) / x.std(dim=2, keepdim=True)
        x = x.view(-1, 1, x.size(1), x.size(2))
        
        x = self.residual_block1(x)
        x = self.pool(x)
        x = self.residual_block2(x)
        x = self.pool(x)
        x = self.residual_block3(x)
        x = self.pool(x)
        x = self.pool(x) # double pool to reduce dimensionality even more
        x = self.dropout_cnn(x)       
        x = x.view(x.size(0), -1, x.size(3))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout_lstm(x)
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze()
        return x
    
    
    

class FCN(nn.Module):
    def __init__(self, input_shape, nb_classes, normalize_features = False):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=128, kernel_size=9, padding='same')
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same')
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, nb_classes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.25)
        self.normalize_features = normalize_features
        
    def forward(self, x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.drop(x)
        x = self.gap(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = x.squeeze()
        return x
    
class ENCODER(nn.Module):
    def __init__(self, input_shape, nb_classes, normalize_features=False):
        super(ENCODER, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_shape[0], out_channels=128, kernel_size=5, stride=1, padding='same'),
            nn.InstanceNorm1d(128),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=11, stride=1, padding='same'),
            nn.InstanceNorm1d(256),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=21, stride=1, padding='same'),
            nn.InstanceNorm1d(512),
            nn.PReLU(),
            nn.Dropout(0.2)
        )
        self.attention_softmax = nn.Softmax(dim=2)
        self.dense = nn.Sequential(
            nn.Linear(256, 128),
        )
        self.instancenorm = nn.InstanceNorm1d(128)
        self.drop = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.output_layer1 = nn.Linear(1280, 256)
        self.output_layer2 = nn.Linear(256, nb_classes)
        self.normalize_features = normalize_features
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x = self.drop(x)
        attention_data, attention_softmax = x[:,:,:256], x[:,:,256:]
        attention_softmax = self.attention_softmax(attention_softmax)
        multiply_layer = attention_softmax * attention_data
        dense_layer = self.dense(multiply_layer)
        norm_layer = self.instancenorm(dense_layer.permute(0, 2, 1))
        flatten_layer = self.flatten(dense_layer)
        flatten_layer = self.drop(flatten_layer)
        output = self.output_layer1(flatten_layer)
        output = self.drop(output)
        output = self.relu(output)
        output = self.output_layer2(output)
        output = output.squeeze()
        return output
    
class TimeLeNet(nn.Module):
    def __init__(self, input_shape, nb_classes, normalize_features = False):
        super(TimeLeNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=5, kernel_size=5, padding='same')
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=5, out_channels=20, kernel_size=5, padding='same')
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.flatten = nn.Flatten()
        self.bn1 = nn.BatchNorm1d(5)
        self.bn2 = nn.BatchNorm1d(20)
        self.fc1 = nn.Linear(20 * int(input_shape[1] / 8), 500)  
        self.fc2 = nn.Linear(500, nb_classes)
        self.relu = nn.ReLU()
        self.normalize_features = normalize_features
        self.drop = nn.Dropout(0.25)
        
    def forward(self, x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = x.squeeze()
        return x

class LSFAN(nn.Module): # https://arxiv.org/pdf/2406.16913
    def __init__(self, input_shape, nb_classes, normalize_features=False):
        super(LSFAN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3, 3), stride=(1, 1), padding = 'same')
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(4, 16, kernel_size=(3, 3), stride=(1, 1), padding = 'same')
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.tap = nn.AdaptiveAvgPool2d((None, 1))  # Temporal Average Pooling
        self.attention = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(int(16 * input_shape[0]), 480)  # Adjust based on the output of TAP
        self.fc2 = nn.Linear(480, nb_classes)
        self.relu = nn.ReLU()
        self.normalize_features = normalize_features
        self.drop = nn.Dropout(0.25)
        
    def forward(self, x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True)
        x = x.view(-1, 1, x.size(1), x.size(2)) 
        x = self.relu(self.conv1(x))
        x = self.drop(x)
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.drop(x)
        x = self.pool2(x)
        x = self.tap(x).squeeze()
        x = x.permute(0, 2, 1)  # Rearrange dimensions for attention
        x, _ = self.attention(x, x, x)
        x = x.permute(0, 2, 1)  # Restore original dimensions
        x = x.flatten(start_dim=1)
        x = self.drop(x)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x).squeeze()
        return x

class MCDCNN(nn.Module):
    def __init__(self, input_shape, nb_classes, normalize_features=False):
        super(MCDCNN, self).__init__()
        n_vars,n_t  = input_shape
        self.n_vars = n_vars
        padding = 'same'  # Adjust padding based on input
        
        self.conv1_layers = nn.ModuleList()
        self.conv2_layers = nn.ModuleList()
        for _ in range(n_vars):
            self.conv1_layers.append(nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=5, padding=padding, stride=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2)
            ))
            self.conv2_layers.append(nn.Sequential(
                nn.Conv1d(8, 8, kernel_size=5, padding=padding, stride=1),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(2),
                nn.Flatten()
            ))
        
        self.fc1 = nn.Linear(1024, 256)  # Adjust the input features to fc1
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, nb_classes)
        self.normalize_features = normalize_features
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.25)
        
    def forward(self, x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True)
        conv2_outputs = []
        for i in range(self.n_vars):
            x_i = x[:, i:i+1, :]  # Select the ith variable
            conv1_out = self.conv1_layers[i](x_i)
            conv2_out = self.conv2_layers[i](conv1_out)
            conv2_outputs.append(conv2_out)
        
        concat_layer = torch.cat(conv2_outputs, dim=1) if self.n_vars > 1 else conv2_outputs[0]
        concat_layer = self.drop(concat_layer)
        fully_connected = self.relu(self.fc1(concat_layer))
        fully_connected = self.drop(fully_connected)
        output = self.relu(self.fc2(fully_connected))
        output = self.drop(output)
        output = self.fc3(output)
        output = output.squeeze()
        return output

class MLP(nn.Module):
    def __init__(self, input_shape, nb_classes, normalize_features=False):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.Linear(np.prod(input_shape), 500)
        self.dropout2 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(500, 500)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(500, 500)
        self.dropout4 = nn.Dropout(0.3)
        self.dense4 = nn.Linear(500, nb_classes)
        self.relu = nn.ReLU()
        self.normalize_features = normalize_features

    def forward(self, x):
        if self.normalize_features:
            x = (x - x.mean(dim=2, keepdim=True))/x.std(dim=2, keepdim=True)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.relu(self.dense1(x))
        x = self.dropout2(x)
        x = self.relu(self.dense2(x))
        x = self.dropout3(x)
        x = self.relu(self.dense3(x))
        x = self.dropout4(x)
        x = self.dense4(x)
        x = x.squeeze()
        return x

class Inception(nn.Module): # from https://github.com/flaviagiammarino/inception-time-pytorch/blob/main/inception_time_pytorch/modules.py#L109
    # original paper https://arxiv.org/abs/1909.04939
    def __init__(self, input_size, filter):
        super(Inception, self).__init__()
        self.bottleneck1 = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )
        
        self.conv10 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=10,
            stride=1,
            padding='same',
            bias=False
        )
        
        self.conv20 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=20,
            stride=1,
            padding='same',
            bias=False
        )
        
        self.conv40 = torch.nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=40,
            stride=1,
            padding='same',
            bias=False
        )
        
        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=3,
            stride=1,
            padding=1,
        )
        
        self.bottleneck2 = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )
        
        self.batch_norm = torch.nn.BatchNorm1d(
            num_features=4 * filters
        )
        
class Residual(torch.nn.Module):
    def __init__(self, input_size, filters):
        super(Residual, self).__init__()
        
        self.bottleneck = torch.nn.Conv1d(
            in_channels=input_size,
            out_channels=4 * filters,
            kernel_size=1,
            stride=1,
            padding='same',
            bias=False
        )

        self.batch_norm = torch.nn.BatchNorm1d(
            num_features=4 * filters
        )
        self.relu = nn.ReLU()
    
    def forward(self, x, y):
        y = y + self.batch_norm(self.bottleneck(x))
        y = self.relu(y)
        return y
class Lambda(torch.nn.Module):
    
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f
    
    def forward(self, x):
        return self.f(x)
    
class InceptionModel(torch.nn.Module):
    def __init__(self, input_size, num_classes, filters, depth):
        super(InceptionModel, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.filters = filters
        self.depth = depth
        
        modules = OrderedDict()
        
        for d in range(depth):
            modules[f'inception_{d}'] = Inception(
                input_size=input_size if d == 0 else 4 * filters,
                filters=filters,
            )
            if d % 3 == 2:
                modules[f'residual_{d}'] = Residual(
                    input_size=input_size if d == 2 else 4 * filters,
                    filters=filters,
                )
        
        modules['avg_pool'] = Lambda(f=lambda x: torch.mean(x, dim=-1))
        modules['linear'] = torch.nn.Linear(in_features=4 * filters, out_features=num_classes)
        
        self.model = torch.nn.Sequential(modules)

    def forward(self, x):
        for d in range(self.depth):
            y = self.model.get_submodule(f'inception_{d}')(x if d == 0 else y)
            if d % 3 == 2:
                y = self.model.get_submodule(f'residual_{d}')(x, y)
                x = y
        y = self.model.get_submodule('avg_pool')(y)
        y = self.model.get_submodule('linear')(y)
        return y

class InceptionTime():
    
    def __init__(self,
                 x,
                 y,
                 filters,
                 depth,
                 models):
        
        '''
        Implementation of InceptionTime model introduced in Ismail Fawaz, H., Lucas, B., Forestier, G., Pelletier,
        C., Schmidt, D.F., Weber, J., Webb, G.I., Idoumghar, L., Muller, P.A. and Petitjean, F., 2020. InceptionTime:
        Finding AlexNet for Time Series Classification. Data Mining and Knowledge Discovery, 34(6), pp.1936-1962.

        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, channels, length) where samples is the number of time series,
            channels is the number of dimensions of each time series (1: univariate, >1: multivariate) and length
            is the length of the time series.

        y: np.array.
            Class labels, array with shape (samples,) where samples is the number of time series.

        filters: int.
            The number of filters (or channels) of the convolutional layers of each model.

        depth: int.
            The number of blocks of each model.
        
        models: int.
            The number of models.
        '''
        
        # Check if GPU is available.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Scale the data.
        self.mu = np.nanmean(x, axis=0, keepdims=True)
        self.sigma = np.nanstd(x, axis=0, keepdims=True)
        x = (x - self.mu) / self.sigma
        
        # Save the data.
        self.x = torch.from_numpy(x).float().to(self.device)
        self.y = torch.from_numpy(y).long().to(self.device)
        
        # Build and save the models.
        self.models = [
            InceptionModel(
                input_size=x.shape[1],
                num_classes=len(np.unique(y)),
                filters=filters,
                depth=depth,
            ).to(self.device) for _ in range(models)
        ]
    
    def fit(self,
            learning_rate,
            batch_size,
            epochs,
            verbose=True):
        
        # Generate the training dataset.
        dataset = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(self.x, self.y),
            batch_size=batch_size,
            shuffle=True
        )
        
        for m in range(len(self.models)):
            
            # Define the optimizer.
            optimizer = torch.optim.Adam(self.models[m].parameters(), lr=learning_rate)
            
            # Define the loss function.
            loss_fn = torch.nn.BCELoss()
            
            # Train the model.
            print(f'Training model {m + 1} on {self.device}.')
            self.models[m].train(True)
            for epoch in range(epochs):
                for features, target in dataset:
                    optimizer.zero_grad()
                    output = self.models[m](features.to(self.device))
                    loss = loss_fn(output, target.to(self.device))
                    loss.backward()
                    optimizer.step()
                    accuracy = (torch.argmax(torch.nn.functional.softmax(output, dim=-1), dim=-1) == target).float().sum() / target.shape[0]
                if verbose:
                    print('epoch: {}, loss: {:,.6f}, accuracy: {:.6f}'.format(1 + epoch, loss, accuracy))
            self.models[m].train(False)
            print('-----------------------------------------')
    
    def predict(self, x):

        # Scale the data.
        x = torch.from_numpy((x - self.mu) / self.sigma).float().to(self.device)
        
        # Get the predicted probabilities.
        with torch.no_grad():
            p = torch.concat([torch.nn.functional.softmax(model(x), dim=-1).unsqueeze(-1) for model in self.models], dim=-1).mean(-1)
        
        # Get the predicted labels.
        y = p.argmax(-1).detach().cpu().numpy().flatten()

        return y
        

def train_model(train_loader, val_loader, model, criterion, optimizer, device, num_epochs=25, model_weights_path=None):
    model = model.to(device)
    
    # Initialize lists to store losses
    epoch_train_losses = []
    epoch_val_losses = []
    batch_train_losses = []
    # Initialize variable to track the lowest validation loss
    best_val_loss = float('inf')
    best_model_params = None  # Placeholder for the best model parameters
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_train_loss = loss.item()
            running_loss += batch_train_loss
            batch_train_losses.append(batch_train_loss)  # Save train loss for the batch
            
            # Print progress update for each batch
            print(f'\rEpoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Train Loss: {batch_train_loss:.4f}', end='')
        
        # Evaluate on validation set after the epoch
        model.eval()
        y_val_true, y_val_pred, epoch_val_loss, epoch_val_R2 = evaluate_model(val_loader, model, device)
        epoch_val_losses.append(epoch_val_loss)  # Save validation loss for the epoch
        
        # Check if the current validation loss is the lowest
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            # Save the best model parameters
            best_model_params = model.state_dict()
            if model_weights_path is not None:
                torch.save(model.state_dict(), model_weights_path)
        
        # Print average training loss and validation loss for the epoch
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_losses.append(epoch_train_loss)
        print(f'\nEpoch {epoch+1}/{num_epochs}, Avg. Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val R^2: {epoch_val_R2:.4f}')

    # After training, load the best model parameters
    model.load_state_dict(best_model_params)
    print(f'\nTraining complete. Best Validation Loss: {best_val_loss:.4f}')
    
    return epoch_train_losses, epoch_val_losses, best_val_loss  # Return the loss lists and best validation loss

def evaluate_model(loader, model, device):
    model.eval()
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            predictions = model(inputs)
            y_pred_list.append(predictions.to('cpu').numpy())
            y_true_list.append(targets.to('cpu').numpy())
    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_true_list)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return y_true, y_pred, mse, r2

# Function to compute predictions and metrics
def compute_predictions_and_metrics(loader, model, device):
    model = model.to(device)
    y_pred_list = []
    y_true_list = []
    model.eval()
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features)
            y_pred_list.append(predictions.to('cpu').numpy())
            y_true_list.append(targets.to('cpu').numpy())
    y_pred = np.concatenate(y_pred_list)
    y_true = np.concatenate(y_true_list)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return y_true, y_pred, mse, r2