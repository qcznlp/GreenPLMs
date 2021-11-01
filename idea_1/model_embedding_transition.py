from torch.functional import Tensor
import transformers
from torch import nn
import torch
from transformers.models.bert.modeling_bert import BertEmbeddings
import numpy as np
from transformers.models.bert.tokenization_bert import BertTokenizer
from dict_cleaning import generate_dict_mapping

def bert_embedding_dict_expansion(new_dict,embedding):
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

def resize_the_model(new_embedding,pretrained_model_path):
    '''
    Resize the model to a bigger dimension at the embedding layer.
    '''
    pre_trained_model = transformers.BertModel.from_pretrained(pretrained_model_path)
    new_vocab_length = new_embedding.num_embeddings
    pre_trained_model.set_input_embeddings(new_embedding)
    pre_trained_model.config.vocab_size = new_vocab_length
    pre_trained_model.vocab_size = new_vocab_length
    return pre_trained_model

new_dict_1 = {}
for k in range(3000):
    new_dict_1[k] = k
    new_dict_1[3000] = 1
    new_dict_1[3001] = 2


# Test pre-trained model
transformers.logging.set_verbosity_error()
model = transformers.BertModel.from_pretrained("E:\\Steve_Zeng_Related\\YLab\\Translation_BERT_project\\code_project\\BERT\\bert-large-cased-whole-word-masking")
print(model)
# tokenizer = BertTokenizer.from_pretrained("E:\\Steve_Zeng_Related\\YLab\\Translation_BERT_project\\code_project\\BERT\\bert-large-cased-whole-word-masking")
# inp = torch.tensor(tokenizer._convert_token_to_id("[unused1]"))
# bert_embedding = model.get_input_embeddings()
embedding = nn.Embedding(1000,1024)
#test_dict = {0:0,1:1,2:2,3:1,4:2}
# input = torch.LongTensor([1,2,3,4])
# print(model)
# print(bert_embedding(input))
# print(bert_embedding(inp))
# print(bert_embedding.num_embeddings)
mapping_dict = generate_dict_mapping("E:\\Steve_Zeng_Related\\YLab\\Translation_BERT_project\\code_project\\idea_1\\translated_dict_4.txt")
mapping_dict_list = list(mapping_dict.keys())
for i in range(len(mapping_dict)):
    mapping_dict[i] = mapping_dict.pop(mapping_dict_list[i])
print(len(mapping_dict))
new_embed = bert_embedding_dict_expansion(new_dict_1, embedding)
print(new_embed)
# new_model = resize_the_model(new_embed,"E:\\Steve_Zeng_Related\\YLab\\Translation_BERT_project\\code_project\\BERT\\bert-large-cased-whole-word-masking")
# print(new_model)