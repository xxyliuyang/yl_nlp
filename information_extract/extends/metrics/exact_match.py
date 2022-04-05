from typing import Dict, List
import torch
from overrides import overrides

from allennlp.training.metrics.metric import Metric


class EMAcc(Metric):

    def __init__(self) -> None:
        self.correct_count = 0.0
        self.total_count = 0.0

    def __call__(self, pred, truth, **kwargs):
        # (batch_size, ..., seq_len, tag_size) -> (batch_size, ..., seq_len)
        pred_id = torch.argmax(pred, dim=-1)
        # (batch_size, ..., seq_len) -> (batch_size, )，把每个 sample 压成一条 seq
        pred_id = pred_id.view(pred_id.size()[0], -1)
        truth = truth.view(truth.size()[0], -1)

        # (batch_size,)，每个元素是 pred 与 truth 之间 tag 相同的数量
        correct_tag_num = torch.sum(torch.eq(truth, pred_id).float(), dim=1)
        # mask掉padding的的数量
        mask_tag_num = torch.sum(torch.eq(truth, -100), dim=1)

        # seq 维上所有 tag 必须正确，所以 correct_tag_num 必须等于 seq 的长度才算一个 correct 的 sample
        truth_tag_num = torch.ones_like(correct_tag_num) * truth.size()[-1] - mask_tag_num
        is_right_for_batch = torch.eq(correct_tag_num, truth_tag_num).float()
        right_count_for_batch = torch.sum(is_right_for_batch)

        self.correct_count += right_count_for_batch
        self.total_count += pred.size()[0]

    def get_metric(self, reset: bool = False):
        if self.total_count > 1e-12:
            accuracy = float(self.correct_count) / float(self.total_count)
        else:
            accuracy = 0.0

        if reset:
            self.reset()

        return accuracy

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_count = 0.0
