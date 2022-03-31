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
            shake_seq_tag += [0 for _ in range(max_length - length)]
        while i < max_length:
            i += 1
            shake_seq_tag += [0 for _ in range((max_length - i))]
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