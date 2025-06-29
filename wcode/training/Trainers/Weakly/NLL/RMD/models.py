import torch
import torch.nn as nn

from wcode.net.CNN.VNet.VNet import Encoder, Decoder


class VNet(nn.Module):
    def __init__(self, params):
        super(VNet, self).__init__()
        self.need_features = params["need_features"]
        self.deep_supervision = params["deep_supervision"]

        self.encoder_params, self.decoder_params = self.get_EnDecoder_params(params)
        self.encoder = Encoder(self.encoder_params)
        self.decoder = Decoder(
            self.decoder_params,
            output_features=self.deep_supervision or self.need_features,
        )

        if len(params["kernel_size"][0]) == 2:
            Conv_layer = nn.Conv2d
        elif len(params["kernel_size"][0]) == 3:
            Conv_layer = nn.Conv3d

        if self.deep_supervision:
            self.prediction_head = nn.ModuleList()
            # we will not do deep supervision on the prediction of bottleneck output feature
            # the prediction_heads are from low to high resolution.
            for i in range(1, len(self.encoder_params["num_conv_per_stage"])):
                self.prediction_head.append(
                    Conv_layer(
                        self.decoder_params["features"][i],
                        params["out_channels"],
                        kernel_size=1,
                        bias=params["need_bias"],
                    )
                )
        else:
            self.prediction_head = Conv_layer(
                self.decoder_params["features"][-1],
                params["out_channels"],
                kernel_size=1,
                bias=params["need_bias"],
            )

    def forward(self, inputs):
        encoder_out = self.encoder(inputs)
        decoder_out = self.decoder(encoder_out)
        if self.deep_supervision:
            outputs = []
            for i in range(len(decoder_out)):
                outputs.append(self.prediction_head[i](decoder_out[i]))
            # we assume that the multi-level prediction ranking ranges from high resolution to low resolution
            if self.need_features:
                net_out = {"feature": encoder_out + decoder_out, "pred": outputs[::-1]}
            else:
                net_out = {"pred": outputs[::-1]}
        else:
            if self.need_features:
                outputs = self.prediction_head(decoder_out[-1])
                net_out = {"feature": encoder_out + decoder_out, "pred": outputs}
            else:
                net_out = {"pred": self.prediction_head(decoder_out)}

        return net_out

    def get_EnDecoder_params(self, params):
        encoder_params = {}
        decoder_params = {}

        encoder_params["in_channels"] = params["in_channels"]
        encoder_params["features"] = params["features"]
        encoder_params["dropout_p"] = params["dropout_p"]
        encoder_params["num_conv_per_stage"] = params["num_conv_per_stage"]
        encoder_params["kernel_size"] = params["kernel_size"]
        encoder_params["pool_kernel_size"] = params["pool_kernel_size"]
        encoder_params["normalization"] = params["normalization"]
        encoder_params["activate"] = params["activate"]
        encoder_params["need_bias"] = params["need_bias"]

        assert (
            len(encoder_params["features"])
            == len(encoder_params["dropout_p"])
            == len(encoder_params["num_conv_per_stage"])
            == len(encoder_params["kernel_size"])
            == (len(encoder_params["pool_kernel_size"]) + 1)
        )

        decoder_params["features"] = params["features"][::-1]
        decoder_params["kernel_size"] = params["kernel_size"][::-1]
        decoder_params["pool_kernel_size"] = params["pool_kernel_size"][::-1]
        decoder_params["dropout_p"] = [0.0 for _ in range(len(params["dropout_p"]))]
        decoder_params["num_conv_per_stage"] = params["num_conv_per_stage"][::-1]
        decoder_params["normalization"] = params["normalization"]
        decoder_params["activate"] = params["activate"]
        decoder_params["need_bias"] = params["need_bias"]

        return encoder_params, decoder_params


class BiNet(nn.Module):
    def __init__(self, params):
        super(BiNet, self).__init__()
        self.net1 = VNet(params)
        self.net2 = VNet(params)

    def forward(self, x):
        net1_out = self.net1(x)
        net2_out = self.net2(x)

        # if isinstance(net1_out["pred"], list) and isinstance(net2_out["pred"], list):
        #     pred = []
        #     for i in range(len(net1_out["pred"])):
        #         pred.append((net1_out["pred"][i] + net2_out["pred"][i]) / 2)
        # else:
        #     pred = (net1_out["pred"] + net2_out["pred"]) / 2

        return {"net1_out": net1_out, "net2_out": net2_out, "pred": net1_out["pred"]}
