import jieba
import os
from transformers import BertTokenizer,BertModel
from tokenization_wobert import WoBertTokenizer

def cut_words_and_transfer(entrance_string,bilingual_dict):
    word_list = jieba.cut(entrance_string)
    trans_list = []
    unknown_list = []
    for word in word_list:
        try:
            trans_word = bilingual_dict[word]
            trans_list.append(trans_word)
        except KeyError:
            unknown_list.append(word)
            trans_list.append("UNK")
    trans_string = " ".join(trans_list)
    return trans_string,unknown_list

def merge_UNK_into_models(unknown_word_list,model_path,vocab_file_path):
    model = BertModel.from_pretrained(model_path)
    tokenizer = WoBertTokenizer(vocab_file_path))

    tokenizer.add_tokens(unknown_word_list)
    model.resize_token_embeddings(len(tokenizer))

    return model,tokenizer

"""
m,t = merge_UNK_into_models(["mscsc"],"E:\\Steve_Zeng_Related\\YLab\\Translation_BERT_project\\code_project\\BERT\\bert-base-uncased-trans")
print(t.tokenize("我喜欢威廉姆斯"))
"""