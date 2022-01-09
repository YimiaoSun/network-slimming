import numpy as np
import torch
import torch.nn as nn


class channel_selection(nn.Module):
    """
    Select channels from the output of BatchNorm2d layer. It should be put directly after BatchNorm2d layer.
    The output shape of this layer is determined by the number of 1 in `self.indexes`.
    """
    # 用我们保留的channel mask，选择出哪个channel对应的是1，哪个channel对应的是0，
    # 然后挑出1对应的channel，筛掉0对应的channel
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correspond sto the channels to be pruned will be set to 0.
        """
        # E.g:
        # 初始，num_channels=16,
        # 现以num_channels=4为例，
        # 则：indexes=nn.Parameter(torch.ones(4))
        # print(indexes) —>
        # Parameter containing:
        # tensor([1., 1., 1., 1.], requires_grad=True)
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (N,C,H,W). It should be the output of BatchNorm2d layer.
        """
        # 令indexes=torch.Tensor([1., 0., 0., 1.])
        # indexes.data.cpu()——> tensor([1., 0., 0., 1.])
        # indexes.data.cpu().numpy() ——> array([1., 0., 0., 1.], dtype=float32)
        # argwhere ——> np.argwhere(indexes_numpy) ——>
        # array([[0],
        #        [3]], dtype=int64) # https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
        # np.squeeze(argwhere)——>array([0, 3], dtype=int64)
        #
        # 问题在于self.indexes.data这个参数从哪里来？如何更新？这才是真正的prune的点
        # 实际上来源于(对于resprune而言)最终将old_modules赋值到new_modules时。具体请看resprune.py
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
        # 如果indexes中只有一个1，squeeze后，会自动去除掉多余维度，变成一个数字，为了保留维度，进行如此额外处理
        # array(3, dtype=int64) 表示第3维是1，其他都是0
        # np.resize(·) ——> array([1], dtype=int64)
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,)) 
        output = input_tensor[:, selected_index, :, :]
        return output
