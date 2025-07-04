import torch
from torch._dynamo import OptimizedModule


# if your weight is from this framework or nnUNet, use this function
def load_pretrained_weights(network, fname, load_all=True, verbose=False):
    saved_model = torch.load(fname, weights_only=False)
    pretrained_dict = saved_model["network_weights"]

    if load_all:
        skip_strings_in_pretrained = []
    else:
        skip_strings_in_pretrained = [
            ".seg_layers.",
            # 'decoder.'
        ]

    mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    # verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, (
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be "
                f"compatible with your network."
            )
            assert model_dict[key].shape == pretrained_dict[key].shape, (
                f"The shape of the parameters of key {key} is not the same. Pretrained model: "
                f"{pretrained_dict[key].shape}; your network: {model_dict[key]}. The pretrained model "
                f"does not seem to be compatible with your network."
            )

    # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}

    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict.keys()
        and all([i not in k for i in skip_strings_in_pretrained])
    }

    model_dict.update(pretrained_dict)

    print(
        "################### Loading pretrained weights from file ",
        fname,
        "###################",
    )
    if verbose:
        print(
            "Below is the list of overlapping blocks in pretrained model and loaded model architecture:"
        )
        for key, value in pretrained_dict.items():
            print(key, "shape", value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)
