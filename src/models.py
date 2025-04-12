import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation='relu', norm=True):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
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
        return self.activation(self.bn(self.conv(x)))


class ColorizationNet(nn.Module):
    def __init__(
        self,
        input_channels=1,
        output_channels=3,
        base_filters=64,
        num_layers=4,
        conv_params=None,
        dropout=0.2,
        use_batchnorm=True,
        activation='relu'
    ):
        super(ColorizationNet, self).__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout

        if conv_params is None:
            conv_params = [{"kernel": 3, "stride": 2, "padding": 1}] * num_layers
        assert len(conv_params) == num_layers, "conv_params must match num_layers"

        self.skip_channels = []

        # ---------- Encoder ----------
        in_ch = input_channels
        filters = base_filters
        for i in range(num_layers):
            p = conv_params[i]
            block = nn.Sequential(
                CNNBlock(in_ch, filters, kernel_size=p["kernel"], stride=p["stride"], padding=p["padding"],
                         activation=activation, norm=use_batchnorm),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
            self.encoder.append(block)
            self.skip_channels.append(filters)
            in_ch = filters
            filters *= 2

        # ---------- Decoder ----------
        filters = filters // 2
        for i in reversed(range(num_layers)):
            p = conv_params[i]
            skip_ch = self.skip_channels[i]
            block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, skip_ch, kernel_size=p["kernel"], stride=p["stride"],
                                   padding=p["padding"], output_padding=1),
                CNNBlock(skip_ch * 2, skip_ch, kernel_size=3, stride=1, padding=1,
                         activation=activation, norm=use_batchnorm),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
            self.decoder.append(block)
            in_ch = skip_ch

        # ---------- Final Output Layer ----------
        self.final_conv = nn.Conv2d(in_ch, output_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        out = x

        # ---------- Encoder ----------
        for layer in self.encoder:
            out = layer(out)
            skips.append(out)

        # ---------- Decoder ----------
        for i, layer in enumerate(self.decoder):
            skip = skips[-(i + 1)]

            # ConvTranspose2d
            out = layer[0](out)

            # Resize if needed to match skip connection
            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(out, size=skip.shape[2:], mode='bilinear', align_corners=False)

            # Concatenate skip connection
            out = torch.cat([out, skip], dim=1)

            # CNNBlock + Dropout
            out = layer[1](out)
            out = layer[2](out)

        return self.final_conv(out)
