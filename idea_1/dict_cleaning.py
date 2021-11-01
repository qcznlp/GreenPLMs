import os
import re
from collections import OrderedDict

def generate_dict_mapping(vocab_file):
    dict_mapping = {}
    with open(vocab_file,encoding="UTF-8") as obj:
        vocab_lines = obj.readlines()
    for i, line in enumerate(vocab_lines):
        line = line.strip()
        if line.startswith("#"):
            dict_mapping[line] = i
        elif "unused" in line:
            dict_mapping[line] = i
        elif "[PAD]" in line or "[UNK]" in line or "[CLS]" in line or "[SEP]" in line or "[MASK]" in line:
            dict_mapping[line] = i
        elif "\\" in line:
            if line == "\\":
                dict_mapping[line] = i
            else:
                role_list = line.split("\\")
                length = len(role_list)
                for k in range(length):
                    try:
                        role = role_list[k].strip()
                        meaning_list = role.split(".")[1].split(",")
                        for meaning in meaning_list:
                            dict_mapping[meaning] = i
                    except IndexError:
                        pass
        elif "\\" not in line and "." in line:
            meaning_string = line.split(".")[1]
            if ";" in meaning_string:
                meaning_list_1 = meaning_string.split(";")
                for potential_big_meaning in meaning_list_1:
                    if "," in potential_big_meaning:
                        meaning_list_2 = potential_big_meaning.split(",")
                        for meaning_1 in meaning_list_2:
                            dict_mapping[meaning_1] = i
                    else:
                        dict_mapping[potential_big_meaning] = i
            else:
                 meaning_list_3 = meaning_string.split(",")
                 for meaning_2 in meaning_list_3:
                     dict_mapping[meaning_2] = i
        elif "过去式" in line or "过去分词" in line or "现在分词" in line:
            word = line.split("的")[0]
            dict_mapping[word] = i
        elif "[" in line:
            try:
                meaning_3 = line.split("]")[1]
                dict_mapping[meaning_3] = i
            except IndexError:
                dict_mapping[line] = i
        else:
            dict_mapping[line] = i
    return dict_mapping

def clean_dict_again(uncleaned_dict):
    vocab = list(uncleaned_dict.keys())
    print(vocab[13757])
    for i, word in enumerate(vocab):
        vocab[i] = re.sub("[\(\[].*?[\)\]]", "", word)
        if "现在式" in word:
            vocab[i] = word.split("的")[0]
        if "现在时" in word:
            vocab[i] = word.split("的")[0]
        if "现在分词" in word:
            vocab[i] = word.split("的")[0]
        if "；" in word and len(word) > 1:
            vocab[i] = word.split("；")[0]
        if ";" in word and len(word) > 1:
            vocab[i] = word.split(";")[0]
    return vocab

def generate_one_on_one_cleaning(vocab_file):
    vocab = []
    with open(vocab_file,encoding="UTF-8") as obj:
        vocab_lines = obj.readlines()
    for line in vocab_lines:
        line = line.strip()
        if "\\" in line:
            first = line.split("\\")[0]
            if "." in first:
                string = first.split(".")[1]
                if "," in string:
                    meaning = string.split(",")[0]
                    vocab.append(meaning)
                else:
                    vocab.append(string)
            else:
                vocab.append(first)
        else:
            if "." in line:
                second_string = line.split(".")[1]
                if "," in second_string:
                    second_meaning = second_string.split(",")[0]
                    vocab.append(second_meaning)
                else:
                    vocab.append(second_string)
            else:
                vocab.append(line)
    return vocab

if __name__ == "__main__":
    mapping = generate_dict_mapping("E:\\Steve_Zeng_Related\\YLab\\Translation_BERT_project\\code_project\\idea_1\\translated_dict_4.txt")
    vocab = generate_one_on_one_cleaning("E:\\Steve_Zeng_Related\\YLab\\Translation_BERT_project\\code_project\\idea_1\\translated_dict_4.txt")
    print(vocab)
    '''
    vocab = clean_dict_again(mapping)
    '''
    with open("vocab_2.txt","a",encoding="UTF-8") as obj:
        for k in vocab:
            k = k.strip()
            obj.write(k+"\n")