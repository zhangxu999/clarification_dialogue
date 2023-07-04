from transformers import AutoTokenizer, AutoModelForMaskedLM,AutoModel
from functools import lru_cache
from transformers import pipeline
import pickle
import os

class EmbeddingModel:
    def __init__(self,device, cache_file, model_name = 'xlm-roberta-base',log_msg=False):
        self.device = device
        self.cache_file = cache_file
        self.log_msg = log_msg
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        
        if os.path.exists(self.cache_file):
            with open(cache_file, 'rb') as f:
                self.mapping = pickle.load(f)
        else:
            self.mapping = {}
        print(f" there is {len(self.mapping)} data in the cache file")
        self.new_item = 0
    
    def _encode(self,text):
        encoded_input = self.tokenizer(text,return_tensors='pt',max_length=512,truncation=True)
        encoded_input = {k:v.to(self.device) for k,v in encoded_input.items()}
        output = self.model(**encoded_input)
        return output.last_hidden_state[0][0].detach().cpu().numpy()
    
    def encode(self,key):
        value = self.mapping.get(key)
        if value is None:
            value = self._encode(key)
            self.mapping[key] = value
            self.new_item += 1
            
        if self.new_item> 400:
            self.save()
        return value
            
    def save(self):
        if self.new_item >0:
            with open(self.cache_file,'wb') as f:
                pickle.dump(self.mapping, f)
        if self.log_msg:
            print(f"wrote cache file done, new_item:{self.new_item}")
        self.new_item = 0

    
    
class Unmasker:
    
    def __init__(self,device, cache_file, model_name = 'xlm-roberta-base', log_msg=False):
        self.device = device
        self.cache_file = cache_file
        self.log_msg = log_msg

        self.model = pipeline('fill-mask', model=model_name, framework='pt', device=device)        
        if os.path.exists(self.cache_file):
            with open(cache_file, 'rb') as f:
                self.mapping = pickle.load(f)
        else:
            self.mapping = {}
        print(f" there is {len(self.mapping)} data in the cache file")
        self.new_item = 0


    def get_mask_words(self,mask_text, top_k):
        origin_words = [(token['token_str'],round(token['score'],3)) for token in self.model(mask_text,top_k=top_k)]
        return origin_words

        
    def get(self, key, top_k):
        value = self.mapping.get((key, top_k))
        if value is None:
            value = self.get_mask_words(key, top_k)
            self.mapping[(key, top_k)] = value
            self.new_item += 1
        if self.new_item> 400:
            self.save()
        return value
    
    def save(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.mapping, f)
        if self.log_msg:
            print(f"wrote cache file done, new_item:{self.new_item}")
        self.new_item = 0
