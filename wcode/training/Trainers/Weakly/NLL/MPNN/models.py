import torch
import torch.nn as nn

from wcode.net.CNN.VNet.VNet import VNet


class LinkNets(nn.Module):
    def __init__(self, params, num_net):
        super(LinkNets, self).__init__()
        self.networks = nn.ModuleList([VNet(params) for _ in range(num_net)])

        # check whether initialize differently
        for i in range(0, len(self.networks) - 1):
            check_idx_lst = list(range(i + 1, len(self.networks)))
            check_results = []
            for c in check_idx_lst:
                different = False
                for param1, param2 in zip(
                    self.networks[i].parameters(), self.networks[c].parameters()
                ):
                    if not torch.allclose(param1, param2):
                        check_results.append(True)
                        different = True
                        break
                if not different:
                    check_results.append(False)
            assert all(check_results)
            
    def forward(self, inputs):
        preds_lst = []
        for net in self.networks:
            preds_lst.append(net(inputs)["pred"])

        return preds_lst


if __name__ == "__main__":
    from wcode.utils.file_operations import open_yaml

    data = open_yaml("./wcode/net/CNN/VNet/VNet_test.yaml")
    model = LinkNets(data["Network2d"], num_net=3)
