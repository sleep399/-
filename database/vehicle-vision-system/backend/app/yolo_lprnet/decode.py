"""LPRNet CTC 贪婪解码，来自参考项目的推理逻辑。"""

from __future__ import annotations

import numpy as np
import torch

from .charset import CHARS


def greedy_decode(prebs, chars: list[str] | None = None) -> str:
    chars = chars or CHARS
    if isinstance(prebs, torch.Tensor):
        prebs = prebs.detach().cpu().numpy()

    if prebs.ndim == 3:
        prebs = prebs[0]

    preb_label = [int(np.argmax(prebs[:, j], axis=0)) for j in range(prebs.shape[1])]

    no_repeat: list[int] = []
    if preb_label:
        pre_c = preb_label[0]
        if pre_c != len(chars) - 1:
            no_repeat.append(pre_c)
        for c in preb_label[1:]:
            if pre_c == c or c == len(chars) - 1:
                if c == len(chars) - 1:
                    pre_c = c
                continue
            no_repeat.append(c)
            pre_c = c

    plate_str = "".join(chars[idx] for idx in no_repeat if 0 <= idx < len(chars))
    return plate_str or "无法识别"
