import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union

from wcode.net.CNN.VNet.VNet import Encoder, Decoder


class FeatureAugmentor(nn.Module):
    def __init__(self, in_channel, out_channel, dim, activation):
        super(FeatureAugmentor, self).__init__()
        self.prototype_proj = nn.Linear(in_channel, out_channel)
        if dim == 2:
            feature_proj_layer = nn.Conv2d
        elif dim == 3:
            feature_proj_layer = nn.Conv3d
        else:
            raise ValueError("Unsupport dim: {}".format(dim))
        self.feature_proj = feature_proj_layer(in_channel, out_channel, kernel_size=1)

        self.fc = feature_proj_layer(in_channel, in_channel, kernel_size=1)

        # Initialize the activation funtion
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "elu":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "selu":
            self.activation = nn.SELU(inplace=True)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = nn.Identity()
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(
        self,
        features: torch.Tensor,
        prototype: torch.Tensor,
        memoried_prototype: Union[torch.Tensor, None] = None,
    ):
        assert prototype.shape[0] == 1
        # project and reshape feature
        ## b, c, z, y, x
        proj_feature = self.feature_proj(features)
        b, c, *_ = proj_feature.shape
        ## b, (zyx), c, 1
        proj_feature_ = proj_feature.reshape(b, c, -1).permute(0, 2, 1).unsqueeze(-1)

        # project and reshape prototype, then aug itself
        ## 1, c
        proj_prototype = self.prototype_proj(prototype)
        ## 1, 1, 1, c
        proj_prototype_ = proj_prototype.unsqueeze(0).unsqueeze(0)
        ## b, (zyx), c, c
        attn = torch.matmul(proj_feature_, proj_prototype_)
        attn_rescale = torch.softmax(attn, dim=-1)
        ## b, c, z, y, x
        prototype_aug = (
            torch.matmul(attn_rescale, proj_prototype_.permute(0, 1, 3, 2))
            .squeeze(-1)
            .permute(0, 2, 1)
            .reshape(*proj_feature.shape)
        )

        if memoried_prototype is not None:
            # project and reshape the selected memoried prototype, then aug itself
            ## 1, c
            proj_m_prototype = self.prototype_proj(memoried_prototype)
            ## 1, 1, 1, c
            proj_m_prototype_ = proj_m_prototype.unsqueeze(0).unsqueeze(0)
            ## b, (zyx), c, c
            attn_m = torch.matmul(proj_feature_, proj_m_prototype_)
            attn_m_rescale = torch.softmax(attn_m, dim=-1)
            ## b, c, z, y, x
            m_prototype_aug = (
                torch.matmul(attn_m_rescale, proj_m_prototype_.permute(0, 1, 3, 2))
                .squeeze(-1)
                .permute(0, 2, 1)
                .reshape(*proj_feature.shape)
            )
        else:
            m_prototype_aug = torch.zeros_like(
                prototype_aug, device=prototype_aug.device
            )

        ## b, c, z, y, x
        prototype_aug_out = self.activation(
            self.fc(torch.cat([prototype_aug, m_prototype_aug], dim=1))
        )

        return prototype_aug_out + features


class UsedVNet(nn.Module):
    def __init__(self, params: dict, num_prototype: int, memory_rate: float):
        super(UsedVNet, self).__init__()
        self.need_features = params["need_features"]
        self.deep_supervision = params["deep_supervision"]

        self.encoder_params, self.decoder_params = self.get_EnDecoder_params(params)

        self.encoder = Encoder(self.encoder_params)
        self.FG = FeatureAugmentor(
            in_channel=self.encoder_params["features"][-1],
            out_channel=self.encoder_params["features"][-1] // 2,
            dim=len(params["kernel_size"][0]),
            activation=self.encoder_params["activate"],
        )

        self.prototype_memory = nn.Parameter(
            torch.zeros(
                num_prototype,
                params["out_channels"] - 1,
                self.encoder_params["features"][-1],
            ),
            requires_grad=False
        )
        self.now_saved_prototype = 0
        self.num_prototype = num_prototype
        self.memory_rate = memory_rate

        self.decoder = Decoder(
            self.decoder_params,
            output_features=self.deep_supervision or self.need_features,
        )

        if len(params["kernel_size"][0]) == 2:
            Conv_layer = nn.Conv2d
            self.dropout = nn.functional.dropout2d
            self.dim = 2
        elif len(params["kernel_size"][0]) == 3:
            Conv_layer = nn.Conv3d
            self.dropout = nn.functional.dropout3d
            self.dim = 3

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
                    )
                )
        else:
            self.prediction_head = Conv_layer(
                self.decoder_params["features"][-1],
                params["out_channels"],
                kernel_size=1,
            )

    def forward(self, x, is_infer=True):
        print(self.prototype_memory)
        main_features = self.encoder(x)
        main_outfeatures = self.decoder(main_features)

        if self.deep_supervision:
            main_outputs = []
            for i in range(len(main_outfeatures)):
                main_outputs.append(self.prediction_head[i](main_outfeatures[i]))

            batch_prototype = self.get_batch_prototypes(
                main_features[-1].detach(), main_outputs[-1].detach()
            )
            seleced_prototype = self.update_memory_bank_and_select_prototype(
                batch_prototype, is_infer
            )

            aux_features = [i for i in main_features]
            aux_features[-1] = self.FG(
                main_features[-1], batch_prototype, seleced_prototype
            )

            aux_outfeatures = self.decoder(aux_features)

            aux_outputs = []
            for i in range(len(aux_outfeatures)):
                aux_outputs.append(self.prediction_head[i](aux_outfeatures[i]))

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

                batch_prototype = self.get_batch_prototypes(
                    main_features[-1].detach(), main_outputs.detach()
                )
                seleced_prototype = self.update_memory_bank_and_select_prototype(
                    batch_prototype, is_infer
                )

                aux_features = [i for i in main_features]
                aux_features[-1] = self.FG(
                    main_features[-1], batch_prototype, seleced_prototype
                )

                aux_outfeatures = self.decoder(aux_features)

                aux_outputs = self.prediction_head(aux_outfeatures[-1])

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

                batch_prototype = self.get_batch_prototypes(
                    main_features[-1].detach(), main_outputs.detach()
                )
                seleced_prototype = self.update_memory_bank_and_select_prototype(
                    batch_prototype, is_infer
                )

                aux_features = [i for i in main_features]
                aux_features[-1] = self.FG(
                    main_features[-1], batch_prototype, seleced_prototype
                )

                aux_outfeatures = self.decoder(aux_features)

                aux_outputs = self.prediction_head(aux_outfeatures)

                net_out = {
                    "pred": main_outputs,
                    "pred_for_train": [
                        main_outputs,
                        aux_outputs,
                    ],
                }

        return net_out

    @torch.no_grad()
    def get_batch_prototypes(self, bottleneck_feat, main_pred_logit):
        """
        We want the prototype to be [classes, channels]
        """
        bs, channel, *_ = main_pred_logit.shape
        prototype_channel = bottleneck_feat.shape[1]

        # prepare data
        ## spatial_size, prototype_channel
        BN_feat_ = (
            bottleneck_feat.reshape(bs, prototype_channel, -1)
            .permute(0, 2, 1)
            .reshape(-1, prototype_channel)
        )
        ## fg_class, spatial_size
        pred_confidence_ = (
            F.interpolate(
                torch.softmax(main_pred_logit, dim=1),
                size=bottleneck_feat.shape[2:],
                mode="trilinear" if self.dim == 3 else "bilinear",
                align_corners=True,
            )
            .reshape(bs, channel, -1)
            .permute(1, 0, 2)
            .reshape(channel, -1)[1:, :]
        )
        prototype = torch.matmul(pred_confidence_, BN_feat_) / pred_confidence_.sum(
            dim=1
        ).unsqueeze(-1)

        return F.normalize(prototype)

    @torch.no_grad()
    def update_memory_bank_and_select_prototype(self, prototype, is_infer):
        if is_infer:
            similarities = F.cosine_similarity(
                prototype.unsqueeze(0), self.prototype_memory, dim=2
            ).mean(dim=1)
            most_similar_idx = torch.argmax(similarities)

            return self.prototype_memory[most_similar_idx]
        else:
            if self.now_saved_prototype < self.num_prototype:
                self.prototype_memory[self.now_saved_prototype] = prototype
                self.now_saved_prototype += 1
                return None
            else:
                similarities = F.cosine_similarity(
                    prototype.unsqueeze(0), self.prototype_memory, dim=2
                ).mean(dim=1)

                least_similar_idx = torch.argmin(similarities)
                self.prototype_memory[least_similar_idx] = (
                    self.memory_rate * self.prototype_memory[least_similar_idx]
                    + (1 - self.memory_rate) * prototype
                )

                most_similar_idx = torch.argmax(similarities)

                return self.prototype_memory[most_similar_idx]

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


class UsedVNet_dropout(nn.Module):
    def __init__(self, params):
        super(UsedVNet_dropout, self).__init__()
        self.need_features = params["need_features"]
        self.deep_supervision = params["deep_supervision"]

        self.encoder_params, self.decoder_params = self.get_EnDecoder_params(params)

        self.encoder = Encoder(self.encoder_params)

        self.decoder = Decoder(
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
                    )
                )
        else:
            self.prediction_head = Conv_layer(
                self.decoder_params["features"][-1],
                params["out_channels"],
                kernel_size=1,
            )

    def forward(self, x):
        main_features = self.encoder(x)

        aux_features = [
            self.dropout(main_features[i], p=self.feature_dropout_rate[i])
            for i in range(len(main_features))
        ]

        main_outfeatures = self.decoder(main_features)
        aux_outfeatures = self.decoder(aux_features)

        if self.deep_supervision:
            main_outputs = []
            for i in range(len(main_outfeatures)):
                main_outputs.append(self.prediction_head[i](main_outfeatures[i]))

            aux_outputs = []
            for i in range(len(aux_outfeatures)):
                aux_outputs.append(self.prediction_head[i](aux_outfeatures[i]))

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
                aux_outputs = self.prediction_head(aux_outfeatures[-1])

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
                aux_outputs = self.prediction_head(aux_outfeatures)

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
        decoder_params["dropout_p"] = [0.0 for _ in range(len(params["dropout_p"]))]
        decoder_params["num_conv_per_stage"] = params["num_conv_per_stage"][::-1]
        decoder_params["normalization"] = params["normalization"]
        decoder_params["activate"] = params["activate"]
        decoder_params["need_bias"] = params["need_bias"]

        return encoder_params, decoder_params
