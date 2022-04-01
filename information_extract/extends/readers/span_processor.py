import re

class SpanPreprocessor:
    def __init__(self, tokenize_func, get_tok2char_span_map_func):
        self._tokenize = tokenize_func
        self._get_tok2char_span_map = get_tok2char_span_map_func

    def _get_ent2char_spans(self, text, ent, ignore_subword_match=True):
        '''
        if ignore_subword_match is true, find entities with whitespace around, e.g. "entity" -> " entity "
        '''
        text_cp = " {} ".format(text) if ignore_subword_match else text

        spans = []
        target_ent = " {} ".format(ent) if ignore_subword_match else ent
        for m in re.finditer(re.escape(target_ent), text_cp):
            if not ignore_subword_match and re.match("\d+", target_ent):  # avoid matching a inner number of a number
                if (m.span()[0] - 1 >= 0 and re.match("\d", text_cp[m.span()[0] - 1])) or \
                        (m.span()[1] < len(text_cp) and re.match("\d", text_cp[m.span()[1]])):
                    continue
            span = [m.span()[0], m.span()[1] - 2] if ignore_subword_match else m.span()
            spans.append(span)
        return spans

    def prepare_span_index(self, record):
        text = record['text']
        # 1.计算entity 的 char span
        ent2char_spans = {}
        for entity in record['entity_list']:
            entity_text = entity['text']
            ent_spans = self._get_ent2char_spans(text, entity_text)
            ent2char_spans[ent_spans] = ent_spans

