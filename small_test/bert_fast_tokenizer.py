from transformers import BertTokenizerFast


if __name__ == '__main__':
    tokenizer = BertTokenizerFast.from_pretrained("resources/bert-base-cased", add_special_tokens=False, do_lower_case=False)
    tokenize = tokenizer.tokenize
    get_tok2char_span_map = lambda text: tokenizer.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]

    text = 'Massachusetts ASTON MAGNA Great Barrington ; also at Bard College , Annandale-on-Hudson , N.Y. , July 1-Aug .'
    print(get_tok2char_span_map(text))
    print(tokenize(text))