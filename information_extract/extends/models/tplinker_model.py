from typing import Dict
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.data.fields import MetadataField
from allennlp.data.fields.text_field import TextFieldTensors
from transformers import AutoModel
from torch import nn
import torch
import json
from information_extract.extends.modules.HandshakingKernel import HandshakingKernel

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

        # 关系种类
        rel2id = json.load(open(rel2id_file))
        rel_size = len(rel2id)

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

        # shaking_hiddens: (batch_size, 1 + ... + seq_len, hidden_size)
        shaking_hiddens = self.handshaking_kernel(last_hidden_state)
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

        # 计算loss
        loss = self.loss_func(ent_shaking_outputs, batch_ent_shaking_tag) \
               + self.loss_func(head_rel_shaking_outputs, batch_head_rel_shaking_tag) \
               + self.loss_func(tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)
        outputs["loss"] = loss
        return outputs

    def loss_func(self, weights=None):
        if weights is not None:
            weights = torch.FloatTensor(weights)
        cross_en = nn.CrossEntropyLoss(weight = weights)
        return lambda pred, target: cross_en(pred.view(-1, pred.size()[-1]), target.view(-1))
