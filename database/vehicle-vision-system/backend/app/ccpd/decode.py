"""CCPD-master/rpnet/rpnetEval.py 解码。"""

from app.ccpd.charset import ads, alphabets, provinces
from app.ccpd.load_data import indices_to_plate_from_list


def label_pred_from_output_y(output_y: list[list[list[float]]]) -> list[int]:
    return [t[0].index(max(t[0])) for t in output_y]


def label_pred_from_logits(char_logits) -> list[int]:
    output_y = [el.data.cpu().numpy().tolist() for el in char_logits]
    return label_pred_from_output_y(output_y)


def indices_to_plate(indices: list[int]) -> str:
    return indices_to_plate_from_list(indices)


def plate_without_province(indices: list[int]) -> str:
    if len(indices) < 7:
        return ""
    return (
        alphabets[indices[1]]
        + ads[indices[2]]
        + ads[indices[3]]
        + ads[indices[4]]
        + ads[indices[5]]
        + ads[indices[6]]
    )


def is_equal(label_gt: list[int], label_pred: list[int]) -> int:
    compare = [1 if int(label_gt[i]) == int(label_pred[i]) else 0 for i in range(7)]
    return sum(compare)
