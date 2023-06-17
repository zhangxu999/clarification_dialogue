from transformers import AutoTokenizer, AutoModelForMaskedLM,AutoModel
from functools import lru_cache

class EmbeddingModel:
    def __init__(self,device, model_name = 'xlm-roberta-base'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
    
    @lru_cache(10000)
    def encode(self,text):
        encoded_input = self.tokenizer(text,return_tensors='pt',max_length=512,truncation=True)
        encoded_input = {k:v.to(self.device) for k,v in encoded_input.items()}
        output = self.model(**encoded_input)
        return output.last_hidden_state[0][0].detach().cpu().numpy()
    
