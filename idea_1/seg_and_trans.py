import jieba
import os
from transformers import BertTokenizer,BertModel
from tokenization_wobert import WoBertTokenizer
jieba.initialize()

def cut_words_and_transfer_via_dict(entrance_string,bilingual_dict):
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

def retrieve_UNK_from_training_materials(training_materials_list,vocab_path):
    with open(vocab_path,"r",encoding="UTF-8") as f:
        contents = f.readlines()
    vocab = [x.strip() for x in contents]
    unk_list = []
    for single_sentence in training_materials_list:
        single_word_list = jieba.cut(single_sentence,HMM = False)
        for single_word in single_word_list:
            if single_word in vocab:
                pass
            else:
                unk_list.append(single_word)
    return unk_list

def merge_UNK_into_models(unknown_word_list,model_path,vocab_file_path):
    model = BertModel.from_pretrained(model_path)
    tokenizer = WoBertTokenizer(vocab_file_path)

    tokenizer.add_tokens(unknown_word_list)
    model.resize_token_embeddings(len(tokenizer))

    return model,tokenizer


"""
m,t = merge_UNK_into_models(["mscsc"],"E:\\Steve_Zeng_Related\\YLab\\Translation_BERT_project\\code_project\\BERT\\bert-base-uncased-trans",
"E:\\Steve_Zeng_Related\\YLab\\Translation_BERT_project\\code_project\\BERT\\bert-base-uncased-trans\\BERT-base-uncased-trans-processed.txt")
print(t.tokenize("我喜欢威廉姆斯"))
"""