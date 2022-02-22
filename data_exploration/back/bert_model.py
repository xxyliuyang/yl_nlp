from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch
from data_exploration.back.data_loader import make_data_loader


class Classifier:
    def __init__(self, label_list, device, model_path):
        self._label_list = label_list
        self._device = device

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._model.to(device)

        self._optimizer = None

        self._dataset = {}
        self._data_loader = {}

    def load_data(self, set_type, examples, batch_size, max_length, shuffle):
        self._dataset[set_type] = examples
        self._data_loader[set_type] = make_data_loader(
            examples=examples,
            label_list=self._label_list,
            tokenizer=self._tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=shuffle)

    def get_optimizer(self, learning_rate, warmup_steps, t_total):
        self._optimizer, self._scheduler = _get_optimizer(
            self._model, learning_rate=learning_rate,
            warmup_steps=warmup_steps, t_total=t_total)

    def train_epoch(self):
        self._model.train()

        for step, batch in enumerate(tqdm(self._data_loader['train'],desc='Training')):
            batch = tuple(t.to(self._device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}

            self._optimizer.zero_grad()
            outputs = self._model(**inputs)
            loss = outputs[0]  # model
            loss.backward()
            self._optimizer.step()
            self._scheduler.step()

    def evaluate(self, set_type):
        self._model.eval()

        preds_all, labels_all = [], []
        data_loader = self._data_loader[set_type]

        for batch in tqdm(data_loader,
                          desc="Evaluating {} set".format(set_type)):
            batch = tuple(t.to(self._device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}

            with torch.no_grad():
                outputs = self._model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
            preds = torch.argmax(logits, dim=1)

            preds_all.append(preds)
            labels_all.append(inputs["labels"])

        preds_all = torch.cat(preds_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)

        return torch.sum(preds_all == labels_all).item() / labels_all.shape[0]


def _get_optimizer(model, learning_rate, warmup_steps, t_total):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    return optimizer, scheduler