import re

class SpanPreprocessor:
    def __init__(self, tokenize_func, get_tok2char_span_map_func):
        self._tokenize = tokenize_func
        self._get_tok2char_span_map = get_tok2char_span_map_func

    def get_ent2char_spans(self, text, entities, ignore_subword_match=True):
        '''
        if ignore_subword_match is true, find entities with whitespace around, e.g. "entity" -> " entity "
        '''
        entities = list(set(entities))
        entities = sorted(entities, key=lambda x: len(x), reverse=True)
        text_cp = " {} ".format(text) if ignore_subword_match else text
        ent2char_spans = {}
        for ent in entities:
            spans = []
            target_ent = " {} ".format(ent) if ignore_subword_match else ent
            for m in re.finditer(re.escape(target_ent), text_cp):
                if not ignore_subword_match and re.match("\d+",target_ent):  # avoid matching a inner number of a number
                    if (m.span()[0] - 1 >= 0 and re.match("\d", text_cp[m.span()[0] - 1])) or \
                            (m.span()[1] < len(text_cp) and re.match("\d", text_cp[m.span()[1]])):
                        continue
                span = [m.span()[0], m.span()[1] - 2] if ignore_subword_match else m.span()
                spans.append(span)
            ent2char_spans[ent] = spans
        return ent2char_spans

    def update_relation_list(self, record, ent2char_spans):
        new_relation_list = []
        for rel in record["relation_list"]:
            subj_char_spans = ent2char_spans[rel["subject"]]
            obj_char_spans = ent2char_spans[rel["object"]]
            for subj_sp in subj_char_spans:
                for obj_sp in obj_char_spans:
                    new_relation_list.append({
                        "subject": rel["subject"],
                        "object": rel["object"],
                        "subj_char_span": subj_sp,
                        "obj_char_span": obj_sp,
                        "predicate": rel["predicate"],
                    })

        record["relation_list"] = new_relation_list

    def update_entity_list(self, record, ent2char_spans):
        if "entity_list" not in record:  # if "entity_list" not in sample, generate entity list with default type
            ent_list = []
            for rel in record["relation_list"]:
                ent_list.append({
                    "text": rel["subject"],
                    "type": "DEFAULT",
                    "char_span": rel["subj_char_span"],
                })
                ent_list.append({
                    "text": rel["object"],
                    "type": "DEFAULT",
                    "char_span": rel["obj_char_span"],
                })
            record["entity_list"] = ent_list
        else:
            ent_list = []
            for ent in record["entity_list"]:
                for char_sp in ent2char_spans[ent["text"]]:
                    ent_list.append({
                        "text": ent["text"],
                        "type": ent["type"],
                        "char_span": char_sp,
                    })
            record["entity_list"] = ent_list

    def get_char2tok_span(self, text):
        tok2char_span = self._get_tok2char_span_map(text)
        char_num = None
        for tok_ind in range(len(tok2char_span) - 1, -1, -1):
            if tok2char_span[tok_ind][1] != 0:
                char_num = tok2char_span[tok_ind][1]
                break
        char2tok_span = [[-1, -1] for _ in range(char_num)]  # [-1, -1] is whitespace
        for tok_ind, char_sp in enumerate(tok2char_span):
            for char_ind in range(char_sp[0], char_sp[1]):
                tok_sp = char2tok_span[char_ind]
                # 因为char to tok 也可能出现1对多的情况，比如韩文。所以char_span的pos1以第一个tok_ind为准，pos2以最后一个tok_ind为准
                if tok_sp[0] == -1:
                    tok_sp[0] = tok_ind
                tok_sp[1] = tok_ind + 1  # 尾部的index+1操作
        return tok2char_span, char2tok_span

    def update_token_span(self, record, char2tok_span):
        def char_span2tok_span(char_span, char2tok_span):
            tok_span_list = char2tok_span[char_span[0]:char_span[1]]
            tok_span = [tok_span_list[0][0], tok_span_list[-1][1]]
            return tok_span

        for rel in record["relation_list"]:
            subj_char_span = rel["subj_char_span"]
            obj_char_span = rel["obj_char_span"]
            rel["subj_tok_span"] = char_span2tok_span(subj_char_span, char2tok_span)
            rel["obj_tok_span"] = char_span2tok_span(obj_char_span, char2tok_span)
        for ent in record["entity_list"]:
            char_span = ent["char_span"]
            ent["tok_span"] = char_span2tok_span(char_span, char2tok_span)

    def add_special_token_length(self, record, special_token_length=1):
        for rel in record["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            rel["subj_tok_span"] = [subj_tok_span[0]+special_token_length, subj_tok_span[1]+special_token_length]
            rel["obj_tok_span"] = [obj_tok_span[0]+special_token_length, obj_tok_span[1]+special_token_length]
        for ent in record["entity_list"]:
            tok_span = ent["tok_span"]
            ent["tok_span"] = [tok_span[0]+special_token_length, tok_span[1]+special_token_length]

    def prepare_span_index(self, record):
        text = record['text']
        # 1.计算entity 的 char span
        entities = [rel["subject"] for rel in record["relation_list"]]
        entities.extend([rel["object"] for rel in record["relation_list"]])
        if "entity_list" in record:
            entities.extend([ent["text"] for ent in record["entity_list"]])
        ent2char_spans = self.get_ent2char_spans(text, entities)

        # 2.更新relation、entity的char span信息
        self.update_relation_list(record, ent2char_spans)
        self.update_entity_list(record, ent2char_spans)

        # 3.计算char span与token span的对应关系
        tok2char_span, char2tok_span = self.get_char2tok_span(text)
        record['tok2char_span'] = tok2char_span

        # 4.更新relation、entity的token span信息
        self.update_token_span(record, char2tok_span)

        # 5.给特殊符号[CLS]加一个token
        self.add_special_token_length(record)


