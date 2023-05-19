from transformers import AutoTokenizer, AutoModelForMaskedLM,AutoModel
from functools import lru_cache

class EmbeddingModel:
    def __init__(self):
        model_name = 'xlm-roberta-base'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    # @lru_cache(3000)
    def encode(self,text):
        encoded_input = self.tokenizer(text,return_tensors='pt')
        output = self.model(**encoded_input)
        return output.last_hidden_state[0][0]
    
