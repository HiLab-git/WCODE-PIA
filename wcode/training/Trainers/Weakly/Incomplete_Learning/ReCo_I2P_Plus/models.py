import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from typing import Union

from wcode.net.CNN.VNet.VNet import VNet, Encoder, Decoder


class FeatEnhanceModule(nn.Module):
    def __init__(
        self,
        proto_inchannel,
        feat_inchannel,
        proto_outchannel,
        dim,
        num_head,
    ):
        super(FeatEnhanceModule, self).__init__()
        # some checks and initialization
        assert feat_inchannel % num_head == 0
        if dim == 2:
            FeatProjLayer = nn.Conv2d
            NormLayer = nn.InstanceNorm2d
        elif dim == 3:
            FeatProjLayer = nn.Conv3d
            NormLayer = nn.InstanceNorm3d
        else:
            raise ValueError("Unsupport dim: {}".format(dim))
        self.dim = dim
        self.num_head = num_head
        self.proto_inchannel = proto_inchannel
        self.proto_outchannel = proto_outchannel

        self.feat_query_proj = FeatProjLayer(
            feat_inchannel, feat_inchannel, kernel_size=1
        )
        self.proto_key_proj = nn.Linear(proto_inchannel, feat_inchannel)
        self.proto_value_proj = nn.Linear(proto_inchannel, feat_inchannel)

        self.feat_outproj = FeatProjLayer(feat_inchannel, feat_inchannel, kernel_size=1)
        self.feat_norm = NormLayer(feat_inchannel)

        if self.proto_outchannel is not None:
            self.proto_outproj = nn.Linear(
                proto_inchannel + feat_inchannel * 2, proto_outchannel
            )

    def forward(
        self,
        feat: torch.Tensor,
        proto: torch.Tensor,
        memoried_proto: Union[torch.Tensor, None] = None,
    ):
        """
        Inputs:
            feat: features to be augmented, b, c1, (z,) y, x
            proto: intra-batch prototype from soft prediction, cls, c2(for lowest dims) or 1, c2*cls
            memoried_proto: EMA inter-batch prototype from proto, 1, c2*cls
        """
        # b, c1, z, y, x
        b, c1, *_ = feat.shape

        # 1. project prototypes
        ## 2, cls, c2 or 2, 1, c2
        if memoried_proto is not None:
            if proto.shape[-1] != self.proto_inchannel:
                proto = proto.reshape(-1, self.proto_inchannel)
            if memoried_proto.shape[-1] != self.proto_inchannel:
                memoried_proto = memoried_proto.reshape(-1, self.proto_inchannel)
            prototypes = torch.stack([proto, memoried_proto], dim=0)
        else:
            if proto.shape[-1] != self.proto_inchannel:
                proto = proto.reshape(-1, self.proto_inchannel)
            prototypes = proto
        ## (2, cls, c1 or 2, 1, c1) -> (1, 2*cls, c1 or 1, 2, c1) assume as (1, 2, c1)
        proto_keys_ = self.proto_key_proj(prototypes)
        proto_values_ = self.proto_value_proj(prototypes)
        proto_keys = proto_keys_.reshape(-1, c1)[None]
        proto_values = proto_values_.reshape(-1, c1)[None]

        # 2. project features
        ## b, c1, z, y, x
        feat_querys = self.feat_query_proj(feat)
        ## b, c1, z, y, x -> b, zyx, c1
        feat_querys = feat_querys.reshape(b, c1, -1).transpose(1, 2)

        # 3. prepare for multi-head attttention
        def multi_head_reshape(x):
            n, s, c = x.shape
            return x.reshape(n, s, self.num_head, c // self.num_head).transpose(1, 2)

        Q = multi_head_reshape(feat_querys)  # b, head, zyx, c1 // num_head
        K = multi_head_reshape(proto_keys)  # 1, head, 2, c1 // num_head
        V = multi_head_reshape(proto_values)  # 1, head, 2, c1 // num_head

        # 4. compute cross-attention score
        ## b, head, zyx, 2
        cross_attn_weight = torch.matmul(Q, K.transpose(-1, -2))
        cross_attn_weight = cross_attn_weight / (c1**0.5)
        cross_attn_weight = F.softmax(cross_attn_weight, dim=-1)

        # 5. use cross-attention score
        ## (b, head, zyx, 2) * (1, head, 2, c1 // num_head) -> b, head, zyx, c1 // num_head
        cross_attn_out = torch.matmul(cross_attn_weight, V)
        ## b, zyx, c1
        cross_attn_out = cross_attn_out.transpose(1, 2).reshape(b, -1, c1)
        ## b, c1, z, y, x
        cross_attn_out = cross_attn_out.transpose(1, 2).reshape(feat.shape)

        # 6. residual connect
        feat_out = feat + self.feat_outproj(cross_attn_out)
        feat_out = self.feat_norm(feat_out)

        if self.proto_outchannel is not None:
            # 9. project
            proto_out = self.proto_outproj(
                torch.concatenate([prototypes, proto_keys_.detach(), proto_values_.detach()], dim=-1)
            )
            return feat_out, proto_out, None

        return feat_out, None, None


class FeatureAugmentor_v4(nn.Module):
    def __init__(self, channels_lst, num_head_lst):
        super(FeatureAugmentor_v4, self).__init__()
        assert len(channels_lst) == len(num_head_lst)

        self.module_lst = nn.ModuleList()
        for i in range(len(channels_lst)):
            self.module_lst.append(
                FeatEnhanceModule(
                    channels_lst[i],
                    channels_lst[i],
                    (channels_lst[i + 1] if i < len(channels_lst) - 1 else None),
                    3,
                    num_head_lst[i],
                )
            )

    def forward(
        self, feat_lst: list, proto: torch.Tensor, memoried_proto: torch.Tensor
    ):
        enhance_feature = []
        for i, f in enumerate(feat_lst):
            f_out, proto, memoried_proto = self.module_lst[i](f, proto, memoried_proto)
            enhance_feature.append(f_out)

        return enhance_feature


class DIVNet_v4(nn.Module):
    def __init__(
        self, params, num_prototype, memory_rate, update_way, select_way, num_head_lst=[1, 1, 2, 4, 8]
    ):
        super(DIVNet_v4, self).__init__()
        torch.set_float32_matmul_precision('high')
        self.need_features = params["need_features"]
        self.deep_supervision = params["deep_supervision"]

        self.encoder_params, self.decoder_params = self.get_EnDecoder_params(params)

        self.encoder = Encoder(self.encoder_params)

        self.decoder = Decoder(
            self.decoder_params,
            output_features=self.deep_supervision or self.need_features,
        )

        self.FA = FeatureAugmentor_v4(self.encoder_params["features"], num_head_lst)

        self.register_buffer(
            "prototype_memory",
            torch.zeros(
                num_prototype,
                params["out_channels"] - 1,
                self.decoder_params["features"][-1],
                requires_grad=False,
            ),
            persistent=True,
        )
        self.prototype_memory: torch.Tensor

        self.proto_init_flag = False
        self.num_prototype = num_prototype
        self.memory_rate = memory_rate
        self.update_way = update_way
        self.select_way = select_way

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
                    )
                )
        else:
            self.prediction_head = Conv_layer(
                self.decoder_params["features"][-1],
                params["out_channels"],
                kernel_size=1,
            )

    def forward(self, x: torch.Tensor, incomplete_label: torch.Tensor = None):
        main_features = self.encoder(x)
        main_outfeatures = self.decoder(main_features)

        if self.deep_supervision:
            main_outputs = []
            for i in range(len(main_outfeatures)):
                main_outputs.append(self.prediction_head[i](main_outfeatures[i]))

            noisy_prototype, noisy_update_flag = self.get_batch_prototypes(
                main_outfeatures[-1].detach(), main_outputs[-1].detach()
            )

            seleced_prototype = self.update_memory_bank_and_select_prototype(
                noisy_prototype,
                noisy_update_flag,
            )

            aux_features = self.FA(main_features, noisy_prototype, seleced_prototype)
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

                noisy_prototype, noisy_update_flag = self.get_batch_prototypes(
                    main_outfeatures[-1].detach(), main_outputs.detach()
                )

                seleced_prototype = self.update_memory_bank_and_select_prototype(
                    noisy_prototype,
                    noisy_update_flag,
                )

                aux_features = self.FA(
                    main_features, noisy_prototype, seleced_prototype
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

                noisy_prototype, noisy_update_flag = self.get_batch_prototypes(
                    main_outfeatures.detach(), main_outputs.detach()
                )

                seleced_prototype = self.update_memory_bank_and_select_prototype(
                    noisy_prototype,
                    noisy_update_flag,
                )

                aux_features = self.FA(
                    main_features, noisy_prototype, seleced_prototype
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
    def get_batch_prototypes(self, features: torch.Tensor, logits: torch.Tensor):
        """
        We want the prototype to be [classes, channels]
        """
        bs, channel, *_ = logits.shape
        proto_channel = features.shape[1]
        _, num_cls, _ = self.prototype_memory.shape

        # prepare data
        ## spatial_size, proto_channel
        BN_feat_ = (
            features.reshape(bs, proto_channel, -1)
            .permute(0, 2, 1)
            .reshape(-1, proto_channel)
        )

        prob_map = torch.softmax(logits, dim=1)
        hard_label = torch.argmax(logits, dim=1, keepdim=True)

        ## fg_class, spatial_size
        pred_confidence_ = (
            prob_map.reshape(bs, channel, -1)
            .permute(1, 0, 2)
            .reshape(channel, -1)[1:, :]
        )

        ## 1, spatial_size
        hard_label_ = hard_label.reshape(bs, 1, -1).permute(1, 0, 2).reshape(1, -1)

        ## fg_class, proto_channel
        prototype = torch.zeros((num_cls, proto_channel), device=features.device)
        update_flag = []
        for i in range(num_cls):
            ## 1, spatial_size
            cls_mask = hard_label_ == (i + 1)
            if torch.any(cls_mask):
                ## 1, proto_channel
                type_origin = pred_confidence_[i][None].dtype
                conf_ = pred_confidence_[i][None].type(torch.float64) * cls_mask
                prototype[i] = (
                    torch.matmul(
                        conf_,
                        BN_feat_.type(torch.float64),
                    )
                    / conf_.sum()
                )[0].type(type_origin)
                update_flag.append(True)
            else:
                update_flag.append(False)

        return F.normalize(prototype), update_flag

    @torch.no_grad()
    def update_memory_bank_and_select_prototype(
        self,
        prototype,
        update_flag,
    ):
        num_proto, *_ = self.prototype_memory.shape

        if self.proto_init_flag is False:
            # init
            for i, flag in enumerate(update_flag):
                if flag:
                    if not all([torch.any(n) for n in self.prototype_memory[:, i]]):
                        # the prototypes of class i still need to be initialized
                        for j in range(num_proto):
                            if not torch.all(self.prototype_memory[j, i]):
                                # is zero
                                self.prototype_memory[j, i] = prototype[i]
                                break
                    else:
                        self.update_prototype_one_cls(prototype[i][None], i)

            # check whether complete the initialization
            whether_have_zero_proto = False
            for i in range(num_proto):
                for j in range(len(update_flag)):
                    if not torch.all(self.prototype_memory[i, j]):
                        whether_have_zero_proto = True

            if not whether_have_zero_proto:
                self.proto_init_flag = True
        else:
            for i, flag in enumerate(update_flag):
                if flag:
                    self.update_prototype_one_cls(prototype[i][None], i)

        selected_prototype = []
        for i, flag in enumerate(update_flag):
            if flag:
                candidate_proto_lst = [
                    self.prototype_memory[n, i][None]
                    for n in range(num_proto)
                    if torch.any(self.prototype_memory[n, i])
                ]
                if len(candidate_proto_lst) != 0:
                    # num, proto_channel
                    candidate_proto = torch.cat(candidate_proto_lst, dim=0)
                    if self.select_way == "most":
                        similarities = F.cosine_similarity(
                            prototype[i][None], candidate_proto
                        )
                        most_similar_idx = torch.argmax(similarities)
                        selected_prototype.append(
                            candidate_proto[most_similar_idx][None]
                        )
                    elif self.select_way == "merge":
                        selected_prototype.append(
                            candidate_proto.mean(dim=0, keepdim=True)
                        )
                    else:
                        raise ValueError(
                            'Unsupport select way: {}. ("most", "merge")'.format(
                                self.select_way
                            )
                        )

        return (
            torch.cat(selected_prototype, dim=1)
            if len(selected_prototype) != 0
            else None
        )

    @torch.no_grad()
    @torch._dynamo.disable
    def update_prototype_one_cls(self, prototype, update_class):
        """
        Inputs:
            prototype: 1, protoype_channel
            update_class: class need to be updated
            update_way: choose the "least" similar one to update, or update "all" based on the similarities
        """
        similarities = F.cosine_similarity(
            prototype, self.prototype_memory[:, update_class]
        )
        if self.update_way == "least":
            # choose the prototype with the least similarity to update
            least_similar_idx = torch.argmin(similarities)
            self.prototype_memory[least_similar_idx, update_class] = (
                self.memory_rate
                * self.prototype_memory[least_similar_idx, update_class]
                + (1 - self.memory_rate) * prototype[0]
            )
        elif self.update_way == "all":
            for i in range(len(similarities)):
                self.prototype_memory[i, update_class] = (
                    self.memory_rate * self.prototype_memory[i, update_class]
                    + (1 - self.memory_rate) * prototype[0]
                )
        else:
            raise ValueError(
                'Unsupport update way: {}. ("least", "all")'.format(self.update_way)
            )

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
