import torch
import torch.nn as nn

from wcode.net.CNN.VNet.VNet import VNet


class BiNet(nn.Module):
    def __init__(self, params):
        super(BiNet, self).__init__()
        self.net1 = VNet(params)
        self.net2 = VNet(params)

    def forward(self, x):
        net1_out = self.net1(x)
        net2_out = self.net2(x)

        if isinstance(net1_out["pred"], list) and isinstance(net2_out["pred"], list):
            pred = []
            for i in range(len(net1_out["pred"])):
                pred.append((net1_out["pred"][i] + net2_out["pred"][i]) / 2)
        else:
            pred = (net1_out["pred"] + net2_out["pred"]) / 2

        return {"net1_out": net1_out, "net2_out": net2_out, "pred": pred}
