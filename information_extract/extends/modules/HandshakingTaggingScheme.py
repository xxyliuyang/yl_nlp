import torch

class HandshakingTaggingScheme(object):
    """docstring for HandshakingTaggingScheme"""
    def __init__(self, rel2id):
        super(HandshakingTaggingScheme, self).__init__()
        self.rel2id = rel2id
        self.id2rel = {ind :rel for rel, ind in rel2id.items()}

        self.tag2id_ent = {
            "O": 0,
            "ENT-H2T": 1, # entity head to entity tail
        }
        self.id2tag_ent = {id_ :tag for tag, id_ in self.tag2id_ent.items()}

        self.tag2id_head_rel = {
            "O": 0,
            "REL-SH2OH": 1, # subject head to object head
            "REL-OH2SH": 2, # object head to subject head
        }
        self.id2tag_head_rel = {id_ :tag for tag, id_ in self.tag2id_head_rel.items()}

        self.tag2id_tail_rel = {
            "O": 0,
            "REL-ST2OT": 1, # subject tail to object tail
            "REL-OT2ST": 2, # object tail to subject tail
        }
        self.id2tag_tail_rel = {id_ :tag for tag, id_ in self.tag2id_tail_rel.items()}


    def get_spots(self, sample):
        '''
        entity spot and tail_rel spot: (span_pos1, span_pos2, tag_id)
        head_rel spot: (rel_id, span_pos1, span_pos2, tag_id)
        '''
        ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots = [], [], []

        for rel in sample["relation_list"]:
            subj_tok_span = rel["subj_tok_span"]
            obj_tok_span = rel["obj_tok_span"]
            ent_matrix_spots.append((subj_tok_span[0], subj_tok_span[1] - 1, self.tag2id_ent["ENT-H2T"]))
            ent_matrix_spots.append((obj_tok_span[0], obj_tok_span[1] - 1, self.tag2id_ent["ENT-H2T"]))

            if  subj_tok_span[0] <= obj_tok_span[0]:
                head_rel_matrix_spots.append((self.rel2id[rel["predicate"]], subj_tok_span[0], obj_tok_span[0], self.tag2id_head_rel["REL-SH2OH"]))
            else:
                head_rel_matrix_spots.append((self.rel2id[rel["predicate"]], obj_tok_span[0], subj_tok_span[0], self.tag2id_head_rel["REL-OH2SH"]))

            # 尾部的index需要-1操作
            if subj_tok_span[1] <= obj_tok_span[1]:
                tail_rel_matrix_spots.append((self.rel2id[rel["predicate"]], subj_tok_span[1] - 1, obj_tok_span[1] - 1, self.tag2id_tail_rel["REL-ST2OT"]))
            else:
                tail_rel_matrix_spots.append((self.rel2id[rel["predicate"]], obj_tok_span[1] - 1, subj_tok_span[1] - 1, self.tag2id_tail_rel["REL-OT2ST"]))

        return ent_matrix_spots, head_rel_matrix_spots, tail_rel_matrix_spots

    def sharing_spots2shaking_tag(self, spots, length):
        '''
        convert spots to shaking seq tag
        spots: [(start_ind, end_ind, tag_id), ], for entiy
        return:
            shake_seq_tag: (length, length)
        '''
        shaking_tag = torch.zeros(length, length).long()
        for sp in spots:
            shaking_tag[sp[0]][sp[1]] = sp[2]
        return shaking_tag.tolist()

    def spots2shaking_tag(self, spots, length):
        '''
        convert spots to shaking seq tag
        spots: [(rel_id, start_ind, end_ind, tag_id), ], for head relation and tail relation
        return:
            shake_seq_tag: (rel_size, length, length)
        '''
        shaking_tag = torch.zeros(len(self.rel2id), length, length).long()
        for sp in spots:
            shaking_tag[sp[0]][sp[1]][sp[2]] = sp[3]
        return shaking_tag.tolist()


    @classmethod
    def matrix_tag2seq_tag(cls, matrix_tag, max_length):
        shake_seq_tag = []
        length = len(matrix_tag)

        for i, row in enumerate(matrix_tag):
            shake_seq_tag += row[i:]
            shake_seq_tag += [-100 for _ in range(max_length - length)] # pad
        while i < max_length:
            i += 1
            shake_seq_tag += [-100 for _ in range((max_length - i))] # pad
        return shake_seq_tag

    @classmethod
    def sharing_shaking_tag4batch(cls, ent_shaking_tags, max_length):
        """
        tags:  (length, length)
        return:
            batch_shake_seq_tag: (batch_size, shaking_seq_len)
        """
        batch_shake_seq_tag = []

        for ent_shaking_tag in ent_shaking_tags:
            shake_seq_tag = cls.matrix_tag2seq_tag(ent_shaking_tag, max_length)
            batch_shake_seq_tag.append(shake_seq_tag)
        return batch_shake_seq_tag

    @classmethod
    def shaking_tag4batch(cls, shaking_tags, max_length):
        '''
        convert spots to batch shaking seq tag
        spots: (rel_size, length, length)
        return:
            batch_shake_seq_tag: (batch_size, rel_size, shaking_seq_len)
        '''

        batch_shake_seq_tag = []
        for shaking_tag in shaking_tags:
            rel_seq_tag = []
            for rel_shaking_tag in shaking_tag:
                shake_seq_tag = cls.matrix_tag2seq_tag(rel_shaking_tag, max_length)
                rel_seq_tag.append(shake_seq_tag)
            batch_shake_seq_tag.append(rel_seq_tag)
        return batch_shake_seq_tag


    # -------------解码------------------------

    def get_sharing_spots_fr_shaking_tag(self, shaking_tag, shaking_ind2matrix_ind):
        '''
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, )
        spots: [(start_ind, end_ind, tag_id), ]
        '''
        spots = []
        for shaking_ind in shaking_tag.nonzero():
            shaking_ind_ = shaking_ind[0].item()
            tag_id = shaking_tag[shaking_ind_]
            matrix_inds = shaking_ind2matrix_ind[shaking_ind_]
            spot = (matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots

    def get_spots_fr_shaking_tag(self, shaking_tag, shaking_ind2matrix_ind):
        '''
        shaking_tag -> spots
        shaking_tag: (rel_size, shaking_seq_len)
        spots: [(rel_id, start_ind, end_ind, tag_id), ]
        '''
        spots = []

        for shaking_inds in shaking_tag.nonzero():
            rel_id = shaking_inds[0].item()
            tag_id = shaking_tag[rel_id][shaking_inds[1]].item()
            matrix_inds = shaking_ind2matrix_ind[shaking_inds[1]]
            spot = (rel_id, matrix_inds[0], matrix_inds[1], tag_id)
            spots.append(spot)
        return spots

    def get_shaking_ind2matrix_ind(self, matrix_size):
        # mapping shaking sequence and matrix
        # e.g. [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        shaking_ind2matrix_ind = [(ind, end_ind) for ind in range(matrix_size) for end_ind in
                                       list(range(matrix_size))[ind:]]

        return shaking_ind2matrix_ind

    def decode_rel_fr_shaking_tag(self,
                                  ent_shaking_tag,
                                  head_rel_shaking_tag,
                                  tail_rel_shaking_tag,
                                  max_length, tok2char_span, text):
        shaking_ind2matrix_ind = self.get_shaking_ind2matrix_ind(max_length)
        rel_list = []
        tok2char_span.append((0,0)) # [CLS]特殊符号的处理

        ent_matrix_spots = self.get_sharing_spots_fr_shaking_tag(ent_shaking_tag, shaking_ind2matrix_ind)
        head_rel_matrix_spots = self.get_spots_fr_shaking_tag(head_rel_shaking_tag, shaking_ind2matrix_ind)
        tail_rel_matrix_spots = self.get_spots_fr_shaking_tag(tail_rel_shaking_tag, shaking_ind2matrix_ind)

        # entity
        head_ind2entities = {}
        for sp in ent_matrix_spots:
            tag_id = sp[2]
            if tag_id != self.tag2id_ent["ENT-H2T"]:
                continue
            if max((sp[0], sp[1])) > len(tok2char_span):
                continue

            char_span_list = tok2char_span[sp[0]:sp[1] + 1]
            char_sp = [char_span_list[0][0], char_span_list[-1][1]]
            ent_text = text[char_sp[0]:char_sp[1]]

            head_key = sp[0]  # take head as the key to entity list start with the head token
            if head_key not in head_ind2entities:
                head_ind2entities[head_key] = []
            head_ind2entities[head_key].append({
                "text": ent_text,
                "tok_span": [sp[0], sp[1] + 1],
                "char_span": char_sp,
            })

        # tail relation
        tail_rel_memory_set = set()
        for sp in tail_rel_matrix_spots:
            rel_id = sp[0]
            tag_id = sp[3]
            if tag_id == self.tag2id_tail_rel["REL-ST2OT"]:
                tail_rel_memory = "{}-{}-{}".format(rel_id, sp[1], sp[2])
                tail_rel_memory_set.add(tail_rel_memory)
            elif tag_id == self.tag2id_tail_rel["REL-OT2ST"]:
                tail_rel_memory = "{}-{}-{}".format(rel_id, sp[2], sp[1])
                tail_rel_memory_set.add(tail_rel_memory)

        # head relation
        for sp in head_rel_matrix_spots:
            rel_id = sp[0]
            tag_id = sp[3]

            if tag_id == self.tag2id_head_rel["REL-SH2OH"]:
                subj_head_key, obj_head_key = sp[1], sp[2]
            elif tag_id == self.tag2id_head_rel["REL-OH2SH"]:
                subj_head_key, obj_head_key = sp[2], sp[1]

            if subj_head_key not in head_ind2entities or obj_head_key not in head_ind2entities:
                # no entity start with subj_head_key and obj_head_key
                continue
            subj_list = head_ind2entities[subj_head_key]  # all entities start with this subject head
            obj_list = head_ind2entities[obj_head_key]  # all entities start with this object head

            # go over all subj-obj pair to check whether the relation exists
            for subj in subj_list:
                for obj in obj_list:
                    tail_rel_memory = "{}-{}-{}".format(rel_id, subj["tok_span"][1] - 1, obj["tok_span"][1] - 1)
                    if tail_rel_memory not in tail_rel_memory_set:
                        # no such relation
                        continue

                    rel_list.append({
                        "subject": subj["text"],
                        "object": obj["text"],
                        "subj_tok_span": [subj["tok_span"][0], subj["tok_span"][1]],
                        "obj_tok_span": [obj["tok_span"][0], obj["tok_span"][1]],
                        "subj_char_span": [subj["char_span"][0], subj["char_span"][1]],
                        "obj_char_span": [obj["char_span"][0], obj["char_span"][1]],
                        "predicate": self.id2rel[rel_id],
                    })
        return rel_list