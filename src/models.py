import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation='relu', norm=True):
        
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        if norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        self.activation = self._get_activation(activation)

    def _get_activation(self, act):
        if act == 'relu':
            return nn.ReLU(inplace=True)
        elif act == 'leakyrelu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act == 'elu':
            return nn.ELU(inplace=True)
        else:
            raise ValueError("Unsupported activation")

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x

class ColorizationNet(nn.Module):
    """
    This model colorizes an image using a U-Net architecture.
    """
    def __init__(
        self,
        input_channels=1,
        output_channels=3,
        base_filters=64,
        num_layers=4,
        dropout=0.2,
        use_batchnorm=True,
        activation='relu', # 'relu', 'leakyrelu', 'elu', 
        enc_feats=False
    ):
        super(ColorizationNet, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.use_batchnorm = use_batchnorm
        self.activation = self._get_activation(activation)
        self.dropout = dropout
        self.enc_feats = enc_feats

        # ---------- Encoder ----------
        in_ch = input_channels
        filters = base_filters
        for i in range(num_layers):
            self.encoder.append(
                CNNBlock(in_ch, filters, kernel_size=3, stride=2, padding=1, activation=activation, norm=use_batchnorm)
            )
            in_ch = filters
            filters *= 2
            # Optional: add dropout
            if dropout > 0:
                self.encoder.append(nn.Dropout(dropout))

            
        

        # ---------- Decoder ----------
        filters = filters // 2
        for i in range(num_layers):
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, filters, kernel_size=3, stride=2, padding=1, output_padding=1),
                CNNBlock(filters, filters, kernel_size=3, stride=1, padding=1, activation=activation, norm=use_batchnorm),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            ))
            in_ch = filters
            filters //= 2
            # Optional: add dropout
            if dropout > 0:
                self.decoder.append(nn.Dropout(dropout))
        

        # ---------- Final Output Layer ----------
        self.final_conv = nn.Conv2d(in_ch, output_channels, kernel_size=1)


    def _get_activation(self, act):
        if act == 'relu':
            return nn.ReLU(inplace=True)
        elif act == 'leakyrelu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act == 'elu':
            return nn.ELU(inplace=True)
        else:
            raise ValueError("Unsupported activation")
        
    

    def forward(self, x):
        enc_feats = []

        # Encoder
        for enc in self.encoder:
            x = enc(x)
            enc_feats.append(x)

        # Decoder
        for i, dec in enumerate(self.decoder):
            x = dec(x)
            # Optional: add skip connections from encoder
            if i < len(enc_feats) and self.enc_feats:
                x = x + F.interpolate(enc_feats[-(i+1)], size=x.shape[2:])

        return torch.sigmoid(self.final_conv(x))  # Output in range [0, 1]
