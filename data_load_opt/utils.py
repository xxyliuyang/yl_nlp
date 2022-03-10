import random
import os
import csv
from transformers import AutoTokenizer

class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, label=None):
        self.guid = guid
        self.text_a = text_a
        self.label = label

class DatasetProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, task_name):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                line = [item.strip() for item in line]
                lines.append(line)
            return lines

class TSVDataProcessor(DatasetProcessor):
    """Processor for dataset to be augmented."""

    def __init__(self, data_dir, skip_header, label_col, text_col):
        self.data_dir = data_dir
        self.skip_header = skip_header
        self.label_col = label_col
        self.text_col = text_col

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "train.tsv")), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "dev.tsv")), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "test.tsv")), "test")

    def get_labels(self, examples):
        """add your dataset here"""
        labels = set()
        for type in ['train', 'dev']:
            for e in examples[type]:
                labels.add(e.label)
        return sorted(labels)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if self.skip_header and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.text_col]
            label = line[self.label_col]
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

def get_data(data_dir, data_seed=159, is_test=True):
    random.seed(data_seed)
    processor = TSVDataProcessor(data_dir=data_dir,
                                 skip_header=True,
                                 label_col=1, text_col=0)
    examples = dict()

    examples['train'] = processor.get_train_examples()
    examples['dev'] = processor.get_dev_examples()
    examples['test'] = processor.get_test_examples()
    if is_test:
        examples['train'] = examples['train'][:30]
        examples['dev'] = examples['dev'][:30]
        examples['test'] = examples['test'][:30]
    return examples, processor.get_labels(examples)

if __name__ == '__main__':
    file_dir = "data_load_opt/data/"
    examples, label_list = get_data(data_dir=file_dir)
    for key, value in examples.items():
        print('#{}: {}'.format(key, len(value)))
