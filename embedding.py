from transformers import AutoTokenizer, AutoModelForMaskedLM,AutoModel

class EmbeddingModel:
    def __init__(self):
        model_name = 'xlm-roberta-base'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
    def encode(self,text):
        encoded_input = self.tokenizer(text,return_tensors='pt')
        output = self.model(**encoded_input)
        return output.last_hidden_state[0][0]