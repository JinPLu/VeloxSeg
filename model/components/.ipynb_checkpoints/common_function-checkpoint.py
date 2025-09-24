import torch

def concat(*args):
    return torch.cat(args, dim=1)

def split_output_channel(output, channels, mappings = lambda x: x):
    
    mapping = mappings
    res = []
    c = 0
    for i in range(len(channels)):
        if isinstance(mappings, list):
            mapping = mappings[i]
        res.append(mapping(output[:, c:c+channels[i]]))
        c += channels[i]
    return res

def check_input(x, in_channels):
    if not isinstance(x, list):
        channels = x.shape[1]
        assert channels == sum(in_channels), f"Input channels should be equal to the sum of in_channels, but got {channels} and {sum(in_channels)}"
        xs = []
        c = 0
        for i in range(len(in_channels)):
            xs.append(x[:, c:c+in_channels[i]])
            c += in_channels[i]
        x = xs
    return x

