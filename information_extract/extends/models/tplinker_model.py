from typing import Dict, Any
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.data.fields import MetadataField
from allennlp.data.fields.text_field import TextFieldTensors
from transformers import AutoModel
from torch import nn
import torch
import json
from overrides import overrides
from information_extract.extends.modules.HandshakingKernel import HandshakingKernel
from information_extract.extends.modules.HandshakingTaggingScheme import HandshakingTaggingScheme
from information_extract.extends.metrics.exact_match import EMAcc

@Model.register("tplinker")
class TPlinkerModel(Model):
    def __init__(
            self,
            vocab: Vocabulary,
            model_name: str,
            rel2id_file: str,
    ):
        super().__init__(vocab)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.cross_en = nn.CrossEntropyLoss(ignore_index=-100)

        # 关系种类
        rel2id = json.load(open(rel2id_file))
        rel_size = len(rel2id)
        self.handshaking_tagger = HandshakingTaggingScheme(rel2id=rel2id)

        self.ent_fc = nn.Linear(hidden_size, 2)
        self.head_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]
        self.tail_rel_fc_list = [nn.Linear(hidden_size, 3) for _ in range(rel_size)]

        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(hidden_size, "cat", "lstm")

        # distance embedding
        self.dist_emb_size = -1
        self.dist_embbedings = None  # it will be set in the first forwarding

        self.ent_add_dist = False
        self.rel_add_dist = False

        # metric
        self._ent_exact_match = EMAcc()
        self._head_exact_match = EMAcc()
        self._tail_exact_match = EMAcc()

    def get_logits(self, hidden_state):
        # shaking_hiddens: (batch_size, 1 + ... + seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(hidden_state)
        shaking_hiddens4ent = shaking_hiddens
        shaking_hiddens4rel = shaking_hiddens

        # 计算logits
        ent_shaking_outputs = self.ent_fc(shaking_hiddens4ent)
        head_rel_shaking_outputs_list = []
        for fc in self.head_rel_fc_list:
            head_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))
        tail_rel_shaking_outputs_list = []
        for fc in self.tail_rel_fc_list:
            tail_rel_shaking_outputs_list.append(fc(shaking_hiddens4rel))
        head_rel_shaking_outputs = torch.stack(head_rel_shaking_outputs_list, dim=1)
        tail_rel_shaking_outputs = torch.stack(tail_rel_shaking_outputs_list, dim=1)

        return ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs

    def get_batch_shaking_tag(self, meta_data: MetadataField, device):
        """
        convert spots to batch shaking seq tag
        因长序列的stack是费时操作，所以写这个函数用作生成批量shaking tag
        如果每个样本生成一条shaking tag再stack，一个32的batch耗时1s，太昂贵
        spots: [(start_ind, end_ind, tag_id), ], for entiy
        return:
            batch_shake_seq_tag: (batch_size, shaking_seq_len)
        """
        length = []
        ent_shaking_tags = []
        head_rel_shaking_tags = []
        tail_rel_shaking_tags = []

        for meta in meta_data:
            length.append(meta['length'])
            ent_shaking_tags.append(meta['ent_shaking_tag'])
            head_rel_shaking_tags.append(meta['head_rel_shaking_tag'])
            tail_rel_shaking_tags.append(meta['tail_rel_shaking_tag'])
        max_length = max(length)

        batch_ent_shaking_tag = HandshakingTaggingScheme.sharing_shaking_tag4batch(ent_shaking_tags, max_length)
        batch_head_rel_shaking_tag = HandshakingTaggingScheme.shaking_tag4batch(head_rel_shaking_tags, max_length)
        batch_tail_rel_shaking_tag = HandshakingTaggingScheme.shaking_tag4batch(tail_rel_shaking_tags, max_length)
        return torch.tensor(batch_ent_shaking_tag, device=device), \
               torch.tensor(batch_head_rel_shaking_tag, device=device), \
               torch.tensor(batch_tail_rel_shaking_tag, device=device)

    def forward(
            self,
            source_tokens: TextFieldTensors,
            meta_data: MetadataField = None) -> Dict[str, torch.Tensor]:
        outputs = dict()

        input_ids = source_tokens["tokens"]["token_ids"]
        token_type_ids = source_tokens["tokens"]["type_ids"]
        attention_mask = source_tokens["tokens"]["mask"]
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        # logits
        ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = self.get_logits(last_hidden_state)
        outputs['batch_pred_ent_shaking_outputs'] = ent_shaking_outputs
        outputs['batch_pred_head_rel_shaking_outputs'] = head_rel_shaking_outputs
        outputs['batch_pred_tail_rel_shaking_outputs'] = tail_rel_shaking_outputs

        if meta_data:
            # transfer_tag
            batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = self.get_batch_shaking_tag(meta_data, input_ids.device)

            # 计算loss
            loss = self.loss_func(ent_shaking_outputs, batch_ent_shaking_tag) \
                   + self.loss_func(head_rel_shaking_outputs, batch_head_rel_shaking_tag) \
                   + self.loss_func(tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)
            outputs["loss"] = loss

            # 计算acc
            outputs["ent_acc"] = self._ent_exact_match(ent_shaking_outputs, batch_ent_shaking_tag)
            outputs["head_acc"] = self._head_exact_match(head_rel_shaking_outputs, batch_head_rel_shaking_tag)
            outputs["tail_acc"] = self._tail_exact_match(tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

            # ---解码---
            self.make_output_human_readable(meta_data, outputs)
        return outputs

    def loss_func(self, pred, target):
        return self.cross_en(pred.view(-1, pred.size()[-1]), target.view(-1))

    @overrides
    def make_output_human_readable(
            self, meta_data: MetadataField, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        batch_pred_ent_shaking_outputs = output_dict['batch_pred_ent_shaking_outputs']
        batch_pred_head_rel_shaking_outputs = output_dict['batch_pred_head_rel_shaking_outputs']
        batch_pred_tail_rel_shaking_outputs = output_dict['batch_pred_tail_rel_shaking_outputs']
        batch_pred_ent_shaking_tag = torch.argmax(batch_pred_ent_shaking_outputs, dim=-1)
        batch_pred_head_rel_shaking_tag = torch.argmax(batch_pred_head_rel_shaking_outputs, dim=-1)
        batch_pred_tail_rel_shaking_tag = torch.argmax(batch_pred_tail_rel_shaking_outputs, dim=-1)

        batch_rel_list = []
        max_length = max([meta['length'] for meta in meta_data])
        for ind in range(batch_pred_ent_shaking_tag.size()[0]):
            pred_ent_shaking_tag = batch_pred_ent_shaking_tag[ind]
            pred_head_rel_shaking_tag = batch_pred_head_rel_shaking_tag[ind]
            pred_tail_rel_shaking_tag = batch_pred_tail_rel_shaking_tag[ind]

            tok2char_span = meta_data[ind]['tok2char_span']
            text = meta_data[ind]['text']

            rel_list = self.handshaking_tagger.decode_rel_fr_shaking_tag(pred_ent_shaking_tag,
                                                                              pred_head_rel_shaking_tag,
                                                                              pred_tail_rel_shaking_tag,
                                                                              max_length, tok2char_span, text)
            batch_rel_list.append(rel_list)
        output_dict['batch_rel_list'] = batch_rel_list
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "ent_acc": self._ent_exact_match.get_metric(reset),
            "head_acc": self._head_exact_match.get_metric(reset),
            "tail_acc": self._tail_exact_match.get_metric(reset),
        }
        return metrics