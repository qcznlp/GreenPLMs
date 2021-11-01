'''
from transformers import BertTokenizer,BertForMaskedLM,logging,AutoModelWithLMHead, AutoTokenizer,pipeline
model_name = 'liam168/trans-opus-mt-en-zh'
model = AutoModelWithLMHead.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModelWithLMHead.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
#tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
translation_pipeline = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer)
    
substitute = translation_pipeline("thank")
print(substitute)
'''
import gzip
with gzip.open('C:\\Users\\QingchengZeng\\Desktop\\en_part_1.txt.gz', 'r') as f:
    file_content = f.readlines()
with open("train.txt","w",encoding="UTF-8") as f:
    for i,sen in enumerate(file_content):
        sentence = sen.decode("utf-8")
        f.write(sentence)