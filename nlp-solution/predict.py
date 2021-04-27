import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import re
import json
import numpy as np
import pandas as pd


def load_model(model_name):
    if model_name =='base':
        try:
            model = joblib.load(f"models/{model_name}/model")
            tokenizer = None
        except ImportError as e:
            raise Exception(e  + " :Cannot find model!")
            
    elif model_name == 'w2v':
        try: 
            model = tf.keras.models.load_model(f"models/{model_name}/model")
            token_file= json.load(open(f"models/{model_name}/token_config.json","r"))
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(token_file)
        except ImportError as e:
            raise Exception(e  + " :Cannot find model!")
    
    elif model_name == 'bert':
        try: 
            from transformers import BertTokenizer
            import tokenizers
            
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)     
            model = tf.keras.models.load_model(f"models/{model_name}/model")
            
            
        except ImportError as e:
            raise Exception(e  + " :Cannot find model!")
            
    else:
        raise NameError("The model is not available!")
            
    return tokenizer, model



def label_class(y):
    if y>0.5:
        return 1
    else:
        return 0
    


    
class NLP_Solution:
    labels = [ '824.account-management.account-access.0',
               '824.account-management.fingerprint-facial-recognition.0',
               '824.company-brand.competitor.0', 
               '824.company-brand.convenience.0',
               '824.company-brand.general-satisfaction.0',
               '824.online-experience.updates-versions.0',
               '824.staff-support.agent-named.0']
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer, self.model = load_model(model_name)
        
    
    def predict(self,text_list):      
        preprocessed = self.preprocessing_text(text_list)
        results = self.model.predict(preprocessed)
        
        if self.model_name in ["w2v","bert"]:
            results = np.array(results).transpose()[0]
            
        results = [[label_class(i) for i in p] for p in results]
        
        output = []
        for r in results:
            temp ={lb: rb for lb,rb in zip(self.labels,r)}
            output.append(temp)
        
        return output
        

    
    def preprocessing_text(self,text_list):
        if self.model_name =="base":
            preprocessed = [re.sub(r'[^\w\s]'," ",text) for text in text_list]
            
        elif self.model_name =="w2v":
            sequences = self.tokenizer.texts_to_sequences(text_list)
            preprocessed = pad_sequences(sequences,maxlen=200)
        
        elif self.model_name =="bert":
            
            def tokenize(text):
                tokenized = self.tokenizer.encode_plus(text,
                                                  truncation=True,
                                                  add_special_tokens = True, 
                                                  max_length = 200, 
                                                  pad_to_max_length = True, 
                                                  return_attention_mask = True,
                                                  return_token_type_ids=False,
                                                  return_tensors='tf')
                
                return tokenized['input_ids'], tokenized['attention_mask']
            
            
            Xids = np.zeros((len(text_list), 200))
            Xmask = np.zeros((len(text_list), 200))

            for i, sentence in zip(range(200),text_list):
                Xids[i, :], Xmask[i, :] = tokenize(sentence)
            
            preprocessed = {'input_ids': Xids, 'attention_mask': Xmask}
        
        else:
            raise NameError("The model is not available!")
    
        
        return preprocessed
            
    
    
    
if __name__=="__main__":
    df = pd.read_csv("../test.csv")
    model = NLP_Solution("base")
    result = model.predict(df.comment)
    