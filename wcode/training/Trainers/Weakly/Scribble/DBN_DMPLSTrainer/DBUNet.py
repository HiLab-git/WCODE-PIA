import torch
import torch.nn as nn

from wcode.net.CNN.VNet.VNet import Encoder, Decoder


class DBUNet(nn.Module):
    def __init__(self, params):
        super(DBUNet, self).__init__()
        self.need_features = params["need_features"]
        self.deep_supervision = params["deep_supervision"]

        self.encoder_params, self.decoder_params = self.get_EnDecoder_params(params)

        self.encoder = Encoder(self.encoder_params)

        self.decoder = Decoder(
            self.decoder_params,
            output_features=self.deep_supervision or self.need_features,
        )
        self.aux_decoder = Decoder(
            self.decoder_params,
            output_features=self.deep_supervision or self.need_features,
        )
        self.feature_dropout_rate = [0.5 for _ in range(len(params["dropout_p"]))]

        if len(params["kernel_size"][0]) == 2:
            Conv_layer = nn.Conv2d
            self.dropout = nn.functional.dropout2d
        elif len(params["kernel_size"][0]) == 3:
            Conv_layer = nn.Conv3d
            self.dropout = nn.functional.dropout3d

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
            self.aux_prediction_head = nn.ModuleList()
            for i in range(1, len(self.encoder_params["num_conv_per_stage"])):
                self.aux_prediction_head.append(
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
            self.aux_prediction_head = Conv_layer(
                self.decoder_params["features"][-1],
                params["out_channels"],
                kernel_size=1,
                bias=params["need_bias"],
            )

    def forward(self, x):
        main_features = self.encoder(x)

        aux_features = [
            self.dropout(main_features[i], p=self.feature_dropout_rate[i])
            for i in range(len(main_features))
        ]

        main_outfeatures = self.decoder(main_features)
        aux_outfeatures = self.aux_decoder(aux_features)

        if self.deep_supervision:
            main_outputs = []
            for i in range(len(main_outfeatures)):
                main_outputs.append(self.prediction_head[i](main_outfeatures[i]))

            aux_outputs = []
            for i in range(len(aux_outfeatures)):
                aux_outputs.append(self.aux_prediction_head[i](aux_outfeatures[i]))

            # we assume that the multi-level prediction ranking ranges from high resolution to low resolution
            if self.need_features:
                net_out = {
                    "feature": [
                        main_features + main_outfeatures,
                        aux_features + aux_outfeatures,
                    ],
                    "pred": main_outputs[::-1],
                    "pred_for_train": [
                        main_outputs[::-1],
                        aux_outputs[::-1],
                    ],
                }
            else:
                net_out = {
                    "pred": main_outputs[::-1],
                    "pred_for_train": [
                        main_outputs[::-1],
                        aux_outputs[::-1],
                    ],
                }
        else:
            if self.need_features:
                main_outputs = self.prediction_head(main_outfeatures[-1])
                aux_outputs = self.aux_prediction_head(aux_outfeatures[-1])

                net_out = {
                    "feature": [
                        main_features + main_outfeatures,
                        aux_features + aux_outfeatures,
                    ],
                    "pred": main_outputs,
                    "pred_for_train": [
                        main_outputs,
                        aux_outputs,
                    ],
                }
            else:
                main_outputs = self.prediction_head(main_outfeatures)
                aux_outputs = self.aux_prediction_head(aux_outfeatures)

                net_out = {
                    "pred": main_outputs,
                    "pred_for_train": [
                        main_outputs,
                        aux_outputs,
                    ],
                }

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
        decoder_params["dropout_p"] = [0. for _ in range(len(params["dropout_p"]))]
        decoder_params["num_conv_per_stage"] = params["num_conv_per_stage"][::-1]
        decoder_params["normalization"] = params["normalization"]
        decoder_params["activate"] = params["activate"]
        decoder_params["need_bias"] = params["need_bias"]

        return encoder_params, decoder_params


if __name__ == "__main__":
    # import time

    # from wcode.utils.file_operations import open_yaml

    # data = open_yaml("./wcode/net/CNN/VNet/VNet_test.yaml")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print("-----VNet2d-----")
    # upl_unet2d = DBUNet(data["Network2d"]).to(device)
    # begin = time.time()
    # with torch.no_grad():
    #     inputs = torch.rand((16, 1, 256, 256)).to(device)
    #     outputs = upl_unet2d(inputs)
    # print("Time:", time.time() - begin)

    # if outputs.__contains__("feature"):
    #     print("Feature:")
    #     for i, feature in enumerate(outputs["feature"]):
    #         print("Data Flow ID:", i)
    #         if isinstance(feature, (list, tuple)):
    #             for f in feature:
    #                 print(f.shape)
    #         else:
    #             print(feature.shape)

    # print("Pred for training outputs:")
    # for i, pred_for_train in enumerate(outputs["pred_for_train"]):
    #     print("Decoder ID:", i)
    #     if isinstance(pred_for_train, (list, tuple)):
    #         for pred in pred_for_train:
    #             print(pred.shape)
    #     else:
    #         print(pred_for_train.shape)

    # print("Pred outputs:")
    # if isinstance(outputs["pred"], (list, tuple)):
    #     for output in outputs["pred"]:
    #         print(output.shape)
    # else:
    #     print(outputs["pred"].shape)
    # total = sum(p.numel() for p in upl_unet2d.parameters())
    # print("Total params: %.3fM" % (total / 1e6))

    from wcode.utils.file_operations import open_yaml

    data = open_yaml("/home/zhl/wlty/WCODE/Configs/LNQ2023.yaml")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DBUNet(data["Network"]).to(device)
    print(sum(p.numel() for p in model.encoder.parameters()))
    print(sum(p.numel() for p in model.decoder.parameters()))
    print(sum(p.numel() for p in model.aux_decoder.parameters()))
