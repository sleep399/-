"""CCPD-master/rpnet/roi_pooling.py::roi_pooling_ims (PyTorch 现代实现)。"""

import torch
import torch.nn.functional as F


def roi_pooling_ims(input_tensor: torch.Tensor, rois: torch.Tensor, size: tuple[int, int] = (16, 8)) -> torch.Tensor:
    # size: (w, h) — 与官方一致
    out_w, out_h = size
    output: list[torch.Tensor] = []
    rois_f = rois.float()
    for i in range(rois.size(0)):
        roi = rois_f[i]
        x1, y1, x2, y2 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        im = input_tensor[i : i + 1, :, y1 : y2 + 1, x1 : x2 + 1]
        output.append(F.adaptive_max_pool2d(im, (out_h, out_w)))
    return torch.cat(output, dim=0)
