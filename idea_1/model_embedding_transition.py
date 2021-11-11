from torch.functional import Tensor
import transformers
from torch import nn
import torch
from transformers.models.bert.modeling_bert import BertEmbeddings
import numpy as np
from transformers.models.bert.tokenization_bert import BertTokenizer
from dict_cleaning import generate_dict_mapping
from transformers import BertModel
import jionlp as jio
import re

with open("E:\\Steve_Zeng_Related\\YLab\\Translation_BERT_project\\code_project\\data\\bert_base_uncased_trans_dict.txt","r",encoding="UTF-8") as f:
    trans_dict_contents = f.readlines()

def containenglish(test_string):
    return bool(re.search('[a-zA-Z]', test_string))

def is_contain_chinese(check_str):
    """
    判断字符串中是否包含中文
    :param check_str: {str} 需要检测的字符串
    :return: {bool} 包含返回True， 不包含返回False
    """
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

without_line = []
for i,item in enumerate(trans_dict_contents):
    item = item.strip()
    if i <= 1995:
        without_line.append(item)
    else:
        if r"\n" in item:
            new_item = item.replace(r"\n"," ")
            without_line.append(new_item)
        else:
            without_line.append(item)

no_propes = []
for i,no_l in enumerate(without_line):
    if i <= 1995:
        no_propes.append(no_l)
    else:
        match = re.search("[\\[\u4e00-\u9fa5\\]]",no_l)
        if "[PAD]" in no_l or "[MASK]" in no_l or "[SEP]" in no_l or "[CLS]" in no_l or "[UNK]" in no_l or "unused" in no_l:
            no_propes.append(no_l)
        elif match:
            no_proper = no_l.split("[")[0]
            no_propes.append(no_proper)
        else:
            no_propes.append(no_l)

no_brackets = []
for i,no_pr in enumerate(no_propes):
    if i <= 1995:
        no_brackets.append(no_pr)
    else:
        if is_contain_chinese(no_pr):
            without_b = jio.remove_parentheses(no_pr)
            no_brackets.append(without_b)
        else:
            no_brackets.append(no_pr)

no_role = []
for no_br in no_brackets:
    no_ro = no_br.replace("pron."," ")
    no_ro = no_ro.replace("n."," ")
    no_ro = no_ro.replace("a."," ")
    no_ro = no_ro.replace("adv."," ")
    no_ro = no_ro.replace("vbl."," ")
    no_ro = no_ro.replace("v."," ")
    no_ro = no_ro.replace("vt."," ")
    no_ro = no_ro.replace("vi."," ")
    no_ro = no_ro.replace("abbr."," ")
    no_ro = no_ro.replace("interj."," ")
    no_ro = no_ro.replace("conj."," ")
    no_ro = no_ro.replace("art."," ")
    no_ro = no_ro.replace("prep."," ")
    no_ro = no_ro.replace("num."," ")
    no_ro = no_ro.replace("aux."," ")
    no_ro = no_ro.replace("pl."," ")
    no_ro = no_ro.strip()
    no_role.append(no_ro)

no_punctuation = []
simple_punctuation = '[;,；，]'
for i,no_rol in enumerate(no_role):
    if i <= 1995:
        no_punctuation.append(no_rol)
    else:
        without_punctuation = re.sub(simple_punctuation, ' ', no_rol)
        no_punctuation.append(without_punctuation)

no_space = []
for no_pu in no_punctuation:
    no_pu = no_pu.strip()
    if " " in no_pu:
        without_s = no_pu.split()
        no_space.append(without_s)
    else:
        no_space.append(no_pu)

no_tense_things = []
for i,no_s in enumerate(no_space):
    if i <= 1995:
        no_tense_things.append(no_s)
    else:
        if type(no_s) == str:
            if containenglish(no_s) and "的" in no_s:
                no_tt = no_s.split("的")[0]
                no_tense_things.append(no_tt)
            elif is_contain_chinese(no_s) and "..." in no_s:
                no_s = no_s.replace("...","")
                no_tense_things.append(no_s)
            else:
                no_tense_things.append(no_s)
        else:
            for i,single in enumerate(no_s):
                if containenglish(single) and "的" in single:
                    del no_s[i]
                elif "..." in single:
                    no_s[i] = single.replace("...","")
                else:
                    pass
            no_tense_things.append(no_s)
                
# print(no_tense_things)
bert_embedding_help_dict = {}
for i, item in enumerate(no_tense_things):
    bert_embedding_help_dict[i] = item
# print(bert_embedding_help_dict)

def bert_embedding_dict_expansion_old(new_dict,embedding):
    '''
    Receives a dictionary and a nn.Embedding object and then generates a new expanded embedding
    '''
    original_vocab_length = embedding.num_embeddings
    dim = embedding.embedding_dim
    original_embedding = embedding.weight
    new_vocab_length = len(new_dict)
    new_word_embeddings = torch.zeros(new_vocab_length,dim)
    vocab = list(new_dict.keys())
    # new_word_embeddings[:original_vocab_length,:] = original_embedding
    for i in range(len(vocab)):
        new_word_embeddings[i,:] = original_embedding[new_dict[vocab[i]],:]
    new_embeddings = nn.Embedding.from_pretrained(new_word_embeddings)
    return new_embeddings

def save_vocab(vocab_list,path):
    with open(path,"w",encoding="UTF-8") as vocab:
        for single_word in vocab_list:
            single_word = single_word.strip()
            vocab.write(single_word + "\n")

def bert_embedding_dict_expansion_new(new_dict,embedding):
    '''
    Receives a dictionary and a nn.Embedding object and then generates a new expanded embedding
    '''
    original_vocab_length = embedding.num_embeddings
    dim = embedding.embedding_dim
    original_embedding = embedding.weight
    # print(original_embedding)

    embedding_expansion_dict = {}
    for k,v in new_dict.items():
        if type(v) == str:
            if v not in embedding_expansion_dict:
                embedding_expansion_dict[v] = str(k)
            else:
                embedding_expansion_dict[v] += "+" + str(k)
        else:
            for single in v:
                if single not in embedding_expansion_dict:
                    embedding_expansion_dict[single] = str(k)
                else:
                    embedding_expansion_dict[single] += "+" + str(k)
    
    new_vocab_length = len(embedding_expansion_dict)
    with torch.no_grad():
        new_word_embeddings = torch.zeros(new_vocab_length,dim,requires_grad=True)
        vocab = list(embedding_expansion_dict.keys())
        for i,word in enumerate(vocab):
            discrimination_string = embedding_expansion_dict[word]
            if "+" not in discrimination_string:
                new_word_embeddings[i,:] = original_embedding[int(discrimination_string),:]
            else:
                discrimination_string_list = [int(x) for x in discrimination_string.split("+")]
                embedding_list = [original_embedding[y,:] for y in discrimination_string_list]
                new_word_embeddings[i,:] = torch.mean(torch.stack(embedding_list))
    # print(new_word_embeddings.requires_grad)
    # print(new_word_embeddings)

    new_embeddings = nn.Embedding.from_pretrained(new_word_embeddings)
    save_vocab(vocab,"E:\\Steve_Zeng_Related\\YLab\\Translation_BERT_project\\code_project\\data\\vocab.txt")
    return new_embeddings

def resize_the_model(new_embedding,pretrained_model_path):
    '''
    Resize the model to a bigger dimension at the embedding layer.
    '''
    pre_trained_model = transformers.BertModel.from_pretrained(pretrained_model_path)
    new_vocab_length = new_embedding.num_embeddings
    pre_trained_model.set_input_embeddings(new_embedding)
    pre_trained_model.config.vocab_size = new_vocab_length
    pre_trained_model.vocab_size = new_vocab_length
    print(pre_trained_model)
    return pre_trained_model

model = BertModel.from_pretrained("E:\\Steve_Zeng_Related\\YLab\\Translation_BERT_project\\code_project\\BERT\\bert-base-uncased")
bert_embeddings = model.get_input_embeddings()
new_embed = bert_embedding_dict_expansion_new(bert_embedding_help_dict,bert_embeddings)
new_model = resize_the_model(new_embed,"E:\\Steve_Zeng_Related\\YLab\\Translation_BERT_project\\code_project\\BERT\\bert-base-uncased")
new_model.save_pretrained("E:\\Steve_Zeng_Related\\YLab\\Translation_BERT_project\\code_project\\BERT\\bert-base-uncased-embedding-changed")