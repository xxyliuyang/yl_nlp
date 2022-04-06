import json
import logging
import os
from typing import Dict, Iterable

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides

logger = logging.getLogger(__name__)
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from information_extract.extends.modules.HandshakingTaggingScheme import HandshakingTaggingScheme
from information_extract.extends.readers.span_processor import SpanPreprocessor

@DatasetReader.register("tplinker")
class TPlinkerReader(DatasetReader):

    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer],
            tokenizer: Tokenizer,
            rel2id_file: str,
            max_tokens: int = 300,
            **kwargs):
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs)
        self._tokenizer = tokenizer
        self._indexers = token_indexers
        self._max_tokens = max_tokens

        self.handshaking_tagger = HandshakingTaggingScheme(rel2id=json.load(open(rel2id_file)))
        get_tok2char_span_map = lambda text: \
            self._tokenizer.tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        self.span_processor = SpanPreprocessor(tokenize_func=self._tokenizer.tokenizer.tokenize, get_tok2char_span_map_func=get_tok2char_span_map)
        self.count = 0

    @overrides
    def _read(self, file_path) -> Iterable[Instance]:
        with open(cached_path(file_path), 'r') as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for row in self.shard_iterable(data_file):
                record = json.loads(row.strip())
                if len(record['text']) == 0:
                    continue
                yield self.text_to_instance(record)

    def text_to_instance(self, record: Dict) -> Instance:
        self.count += 1
        fields: Dict[str, Field] = {}

        # 1, 构建输入
        text = record['text']
        tokenized_source = self._tokenizer.tokenize(text)
        if self._max_tokens and len(tokenized_source) > self._max_tokens:
            tokenized_source = tokenized_source[: self._max_tokens]
        tokenized_source = self._tokenizer.add_special_tokens(tokenized_source)
        source_field = TextField(tokenized_source, self._indexers)
        fields["source_tokens"] = source_field

        # 2，构建目标
        length = len(tokenized_source)
        ent_shaking_tag, head_rel_shaking_tag, tail_rel_shaking_tag, tok2char_span, = self.prepare_tag_matrix(record, length)
        meta_field = MetadataField({
            "length": length,
            "text": text,
            "tok2char_span": tok2char_span,
            "ent_shaking_tag": ent_shaking_tag,
            "head_rel_shaking_tag": head_rel_shaking_tag,
            "tail_rel_shaking_tag": tail_rel_shaking_tag,
        })
        fields["meta_data"] = meta_field

        # debug
        if self.count < 5:
            logger.info("debug:text={}.\n source tokens={}".format(text, str(tokenized_source)))

        return Instance(fields)

    def prepare_tag_matrix(self, record: Dict, length: int):
        # 1.计算实体、关系的token span
        self.span_processor.prepare_span_index(record)

        # 2.计算spots
        ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = self.handshaking_tagger.get_spots(record)

        # 3.计算tag矩阵
        ent_shaking_tag = self.handshaking_tagger.sharing_spots2shaking_tag(ent_matrix_spots, length)
        head_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag(head_rel_matrix_spots, length)
        tail_rel_shaking_tag = self.handshaking_tagger.spots2shaking_tag(tail_rel_matrix_spots, length)
        return ent_shaking_tag, head_rel_shaking_tag, tail_rel_shaking_tag, record['tok2char_span']

if __name__ == '__main__':
    from allennlp.data.tokenizers import PretrainedTransformerTokenizer
    from allennlp.data.token_indexers import PretrainedTransformerIndexer
    token_indexers = PretrainedTransformerIndexer("../resources/roberta-base", "tokens")
    tokenizer = PretrainedTransformerTokenizer("../resources/roberta-base", add_special_tokens=False)

    filename = "./data/nyt/valid.json"
    reader = TPlinkerReader(
        token_indexers={"tokens": token_indexers},
        tokenizer=tokenizer,
        rel2id_file="./data/nyt/rel2id.json"
    )

    for line in reader._read(filename):
        print(line)
        break