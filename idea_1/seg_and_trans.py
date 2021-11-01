import jieba
import os
from facebook_dict_cleaning import generate_dict_mapping_from_dictionary_zh_en
from transformers import BertTokenizer,BertModel
from word2word import Word2word

jieba.enable_paddle()

generated_dict = generate_dict_mapping_from_dictionary_zh_en("zh","en")

def cut_words_and_transfer(entrance_string):
    zh2en = Word2word("zh_cn","en")
    word_list = jieba.cut(entrance_string,use_paddle=True)
    trans_list = []
    unknown_list = []
    for word in word_list:
        try:
            trans_word = zh2en(word)[0]
            trans_list.append(trans_word)
        except KeyError:
            unknown_list.append(word)
            trans_list.append("UNK")
    trans_string = " ".join(trans_list)
    return trans_string,unknown_list

def merge_UNK_into_models(unknown_word_list,model_path,new_path):
    model = BertModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    tokenizer.add_tokens(unknown_word_list)
    model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained(new_path)
    tokenizer.save_pretrained(new_path)
