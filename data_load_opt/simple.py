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
        self.num_training_instances = len(features)

    def __len__(self):
        return int(self.num_training_instances)

    def __getitem__(self, idx):
        idx = idx % len(self.features)
        feature = self.features[idx]
        return feature


def convet_text_feature(examples):
    tokenizer = AutoTokenizer.from_pretrained(tokenize_model)
    features = []
    for example in examples['train']:
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.text_a, add_special_tokens=True))
        features.append(ids)

    # pad
    length = [len(ids) for ids in features]
    max_length = max(length)
    for i, feature in enumerate(features):
        features[i] = feature + [tokenizer.pad_token_id for _ in range(max_length-len(feature))]
    print(max_length)
    return features

if __name__ == '__main__':
    file_dir = "data_load_opt/data/"
    tokenize_model = "./resources/roberta-base"
    examples, label_list = get_data(data_dir=file_dir)

    # 构建feature
    features = convet_text_feature(examples)

    # 构建dataset、dataloader、sampler
    train_dataset = ClassifyDataset(features, examples)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=4, collate_fn=torch.Tensor)

    # 获取训练数据
    for step, batch in enumerate(train_dataloader):
        print(batch.shape)



