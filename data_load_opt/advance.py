"""以文本分类为例，简单的数据加载"""
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from data_load_opt.utils import get_data
from transformers import AutoTokenizer


class ClassifyDataset(Dataset):
    def __init__(
            self, features, examples):
        self.features = features
        self.examples = examples
        self.max_len = 50
        self.num_training_instances = len(features)

    def __len__(self):
        return int(self.num_training_instances)

    def _trunk(self, feature):
        if len(feature) > self.max_len - 1:
            feature = feature[:self.max_len - 1]
        feature = feature + [feature[-1]]
        return feature

    def __getitem__(self, idx):
        idx = idx % len(self.features)
        feature = self.features[idx]
        feature = self._trunk(feature)
        return feature


def convet_text_feature(examples):
    features = []
    for example in examples['train']:
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.text_a, add_special_tokens=True))
        features.append(ids)

    # no pad
    length = [len(ids) for ids in features]
    max_length = max(length)
    print(max_length)
    return features

def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    max_len = max([len(x) for x in batch])
    for feature in batch:
        feature = feature + [tokenizer.pad_token_id for _ in range(max_len-len(feature))]
        batch_tensors.append(feature)
    return torch.tensor(batch_tensors, dtype=torch.long)


if __name__ == '__main__':
    file_dir = "data_load_opt/data/"
    tokenize_model = "./resources/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(tokenize_model)
    examples, label_list = get_data(data_dir=file_dir)

    # 构建feature
    features = convet_text_feature(examples)

    # 构建dataset、dataloader、sampler
    train_dataset = ClassifyDataset(features, examples)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=4, collate_fn=batch_list_to_batch_tensors)

    # 获取训练数据
    for step, batch in enumerate(train_dataloader):
        print(batch.shape)



