from os import path
import torch.utils.data as Data
import torch
from random import randint,shuffle,randrange,random
import re
import transformers
from transformers.training_args import TrainingArguments
from substitution import *
import torch.optim as optim
import torch.nn as nn
from transformers import LineByLineTextDataset,logging,DataCollatorForLanguageModeling,Trainer,TrainingArguments

logging.set_verbosity_warning()

'''
text = ('Hello, how are you? I am Romeo.\n' # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n' # J
    'Nice meet you too. How are you today?\n' # R
    'Great. My baseball team won the competition.\n' # J
    'Oh Congratulations, Juliet\n' # R
    'Thank you Romeo' # J
)

bilingual_dict = {"hello":"你好","how":"如何","are":"是","you":"你","i":"我","am":"是","romeo":"罗密欧","my":"我的","name":"名字",
"is":"是","juliet":"朱丽叶","nice":"很好的","to":"向","meet":"遇到","too":"也","today":"今天","great":"很棒的","baseball":"棒球",
"team":"队伍","won":"赢","the":"某个","competition":"比赛","oh":"噢","congratulations":"祝贺","thank":"谢谢"}
'''
corpus_path = ""
save_substituted_corpus_path = ""
BERT_model_path = ""
output_path  = ""

sentences = generate_training_materials(corpus_path)
full_path,new_vocab = a_bunch_of_translation_substitutions(sentences,save_substituted_corpus_path)
tokenizer,model = resize_the_vocabulary(BERT_model_path,new_vocab)

dataset = LineByLineTextDataset(tokenizer = tokenizer,file_path = full_path,block_size = 512)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=output_path,
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=32,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)

trainer.train()
trainer.save_model(output_path)
