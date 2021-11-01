import os
import random
from transformers import BertTokenizer,BertForMaskedLM,logging,AutoModelWithLMHead, AutoTokenizer,pipeline
import re
import string

remove = str.maketrans('','',string.punctuation) 
logging.set_verbosity_warning()

def generate_training_materials(corpus_path):
    cleaned_corpus = []
    with open(corpus_path) as f:
        contents = f.readlines()
    for i,item in enumerate(contents):
        lower = item.lower()
        senten = lower.translate(remove)
        cleaned_corpus.append(senten)
    return cleaned_corpus

def do_translation_substitution(sen):
    sen_list = sen.split(" ")
    sen_length = len(sen_list)
    selected_index = random.randint(0,sen_length-1)
    selected_word = sen_list[selected_index]
    model = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
    translation_pipeline = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer)
    
    substitute = translation_pipeline(selected_word)[0]['translation_text']
    substitute = substitute.replace(" ","")
    del sen_list[selected_index]
    for number, char in enumerate(substitute):
        sen_list.insert(selected_index + number,char)
        # sen_list[selected_index] = substitute
    substituted_sentence = " ".join(sen_list)

    return substitute, substituted_sentence

def do_substitution(sen,lang_dict,from_lang = "en",to_lang = "zh"):
    sen_list = sen.split(" ")
    sen_length = len(sen_list)
    selected_index = random.randint(0,sen_length-1)
    selected_word = sen_list[selected_index]
    
    try:
        substitute = lang_dict[selected_word]
        del sen_list[selected_index]
        for number, char in enumerate(substitute):
            sen_list.insert(selected_index + number,char)
        # sen_list[selected_index] = substitute
        substituted_sentence = " ".join(sen_list)
    except KeyError:
        do_substitution(sen,lang_dict)
    
    return substitute, substituted_sentence

def a_bunch_of_substitutions(corpora_list,language_dict,path):
    sub_word_list, sub_sen_list, new_vocab = [],[],[]
    for item in corpora_list:
        sub_word, sub_sentence = do_substitution(item,language_dict)
        sub_word_list.append(sub_word)
        sub_sen_list.append(sub_sentence)
    for word in sub_word_list:
        characters = list(word)
        for character in characters:
            new_vocab.append(character)
    full_path = os.path.join(path,"substituted_corpus.txt")
    with open(full_path,"w",encoding="UTF-8") as f:
        for i in sub_sen_list:
            f.write(i+"\n")
    return full_path,new_vocab

def a_bunch_of_translation_substitutions(corpora_list,path):
    sub_word_list, sub_sen_list, new_vocab = [],[],[]
    for item in corpora_list:
        sub_word, sub_sentence = do_translation_substitution(item)
        sub_word_list.append(sub_word)
        sub_sen_list.append(sub_sentence)
    for word in sub_word_list:
        characters = list(word)
        for character in characters:
            new_vocab.append(character)
    full_path = os.path.join(path,"substituted_corpus.txt")
    with open(full_path,"w",encoding="UTF-8") as f:
        for i in sub_sen_list:
            f.write(i+"\n")
    return full_path,new_vocab

def resize_the_vocabulary(pre_trained_model_path,new_vocab_list):
    tokenizer = BertTokenizer.from_pretrained(pre_trained_model_path)
    model = BertForMaskedLM.from_pretrained(pre_trained_model_path)

    tokenizer.add_tokens(new_vocab_list)
    model.resize_token_embeddings(len(tokenizer))

    return tokenizer,model
