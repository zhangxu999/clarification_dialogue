from collections import defaultdict,namedtuple
import gzip
import json
import random
from itertools import combinations
from transformers import AutoTokenizer, AutoModelForMaskedLM
import spacy

nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

Question = namedtuple('question',['text','target','substitutes'])
class OriginAgent:
    def __init__(self,file_path='data/swords-v1.1_dev.json.gz'):
        with gzip.open(file_path, 'r') as f:
            swords = json.load(f)        

        for key,value in swords['substitute_labels'].items():
            swords['substitutes'][key]['labels'] = value
            swords['substitutes'][key]['label_score'] = value.count('TRUE')/len(value)


        for key,value in swords['substitutes'].items():
            target_id = value['target_id']
            if 'substitutes' not in swords['targets'][target_id]:
                swords['targets'][target_id]['substitutes'] = []
            swords['targets'][target_id]['substitutes'].append(value)

        for key,value in swords['targets'].items():
            context_id = swords['targets'][key]['context_id']
            if 'targets' not in swords['contexts'][context_id]:
                swords['contexts'][context_id]['targets'] = []
            swords['contexts'][context_id]['targets'].append(value)
            
        self.contexts = swords['contexts']
        
        self.context_ids = list(self.contexts.keys())
    
    
    def sample(self,context_id=None):
        if context_id is None:
            context_id = random.choice(self.context_ids)
        context = self.contexts[context_id]['context']
        
        target = self.contexts[context_id]['targets'][0]
        target_text = target['target']
        
        
        substitutes = [(sub['substitute'],sub['label_score']) for sub in target['substitutes']]
        sorted_subs = sorted(substitutes,key=lambda x:x[1],reverse=True)

        # labels = [self.swords['substitute_labels'][sid] for sid in self.tid_to_sids[target_id]]
        return Question(context,target_text , sorted_subs)
    
    def answer(self):
        context = self.sample()[0]
        return context[:20]
        
        
from enum import Enum

# 定义一个枚举类型
class Action(Enum):
    NO_ACTION = 0
    CONFIRM = 1
    OPTION = 2
    EXPLAIN = 3


class DialougeEnv:
    
    def __init__(self,agent,model,mask_model):
        
        class ActionSpace:
            '''
            0:  No action (will answer the question directly)
            1: Confirm   (The word "set" is not clear to me. Do you mean something like ___ by this?)
            2: Options   (he word "set" is not clear to me. Do you mean something like i)___, ii)___ or iii)___ by this?)
            3: More information (The word "set" is not clear to me. Could you please provide me more context?
            ''' 
            actions = list(range(4))
            n = len(actions)
            def sample():
                return random.sample(ActionSpace.actions,1)[0]
        self.action_space = ActionSpace
        
        
        self.OA = agent
        self.model = model
        self.mask_model = mask_model
        
        self.history = []
        self.substitutes = None
        self.target = None
        self.contexts = agent.contexts
        
        self.understand_score = 0
        
        self.reward_table = {
        Action.NO_ACTION.value:{
            True:{
                'answer':'',
                'reward':2,
            },
            False:{
                'answer':" you misunderstdood my words, I mean...",
                'reward': -2
        }},
        Action.CONFIRM.value:{
            True:{
                'answer':'Yes, it is',
                'reward': 1.5
            },
            False:{
                'answer':"No, it is not ",
                'reward': -1.5
        }},
        Action.OPTION.value:{
            True:{
                'answer':None,
                'reward':1,
            },
            False:{
                'answer':" none of these",
                'reward': -1
        }},
        Action.EXPLAIN.value:{
            True:{
                'answer':"the explain content",
                'reward':0.5,
            },
            False:{
                'answer':"the explain content",
                'reward': -0.5  ##  -1
        }}
        }

    
    def should_no_action(self,option_words):
        words = [word for word,score in option_words]
        return self.target in words
        
    def should_confirm(self,option_words):
        words = [word for word,score in option_words]
        sub_words = [word for word,score in self.substitutes if score >0.1]
        is_in_subwords = words[0] in sub_words
        gap = option_words[0][1] - option_words[1][1]
        return is_in_subwords and gap>0.4
            
    
    def should_opt(self,option_words):
        sub_words = [word for word,score in self.substitutes if score > 0]
        words = [word for word,score in option_words]
        return bool(set(words) & set(sub_words))
        
    
    def should_explain(self,option_words):
        sub_words = [word for word,score in self.substitutes if score > 0]
        words = [word for word,score in option_words]
        return not bool(set(words) & set(sub_words))
 
    
        
    def user_utterance(self,action,target,option_words):
        
         
        reward_table = self.reward_table
        if action == Action.NO_ACTION.value:
            is_right_action = self.should_no_action(option_words)
            answer_reward = reward_table[action][is_right_action]
        elif action == Action.CONFIRM.value:
            is_right_action = self.should_confirm(option_words)
            answer_reward = reward_table[action][is_right_action]
        elif action == Action.OPTION.value:
            is_right_action = self.should_opt(option_words)
            answer_reward = reward_table[action][is_right_action]
            answer = f'{random.choice(option_words)[0]}' if is_right_action else 'none of these'
            answer_reward['answer'] = answer
        else:
            is_right_action = self.should_explain(option_words)
            answer_reward = reward_table[action][is_right_action]
        
        return answer_reward
        
        
    def generate_prompt(self,text,target,substitutes):
        mask_text_doc = nlp(text)
        mask_text = ''
        found = False
        for sentence in mask_text_doc.sents:
            # print(sentence.text)
            if (target in sentence.text) and (found is False):
                found = True
                words = [target,*substitutes[:5], '<mask>']
                sub_sentence_text = ".".join([sentence.text.replace(target,w,1) for w in words])
                mask_text += sub_sentence_text
            else:
                mask_text += sentence.text
            if len(tokenizer(mask_text)['input_ids'])>512:
                mask_text = ''
                for sentence in mask_text_doc.sents:
                    # print(sentence.text)
                    if (target in sentence.text):
                        parts = sentence.text.split(",")
                        mask_idx = [i for i,p in enumerate(parts) if (target in p)][0]
                        words = [target,*substitutes[:5],'<mask>']
                        sub_sentence_text = ",".join([parts[mask_idx].replace(target,w,1) for w in words])
                        parts[mask_idx] = sub_sentence_text
                        mask_text += ' '.join(parts)
                    else:
                        mask_text += sentence.text
        while len(tokenizer(mask_text)['input_ids'])>500:
            mask_text = mask_text[:len(mask_text)-10]
            if '<mask>' not in mask_text:
                mask_text = ('<mask>' + mask_text)

        return mask_text
        
        
    
    def get_option_words_by_llm(self):
        # state,info = test_env.reset()
        # h0 =self.history[0]
        
        h0 = self.history[0]
        mask_text = self.generate_prompt(h0.text,h0.target,[w for w,score in h0.substitutes])
        try:
            optional_words = self.mask_model(mask_text)
            new_words = [(token['token_str'],round(token['score'],3)) for token in optional_words]
        except Exception as e:
            print(mask_text)
            print(optional_words)
        return new_words
    
    def bot_utterance(self,action,target):
        option_words = self.get_option_words_by_llm()
        if action == Action.NO_ACTION.value:
            text = ''
        elif action == Action.CONFIRM.value:
            word,score = option_words[0]
            text = f"The word {target} is not clear to me. Do you mean something like {word} by this?"
        elif action == Action.OPTION.value:
            words = [word for word,score in option_words]
            text = f"The word {target} is not clear to me. Do you mean something like {','.join(words)} ?"
        elif action == Action.EXPLAIN.value:
            text = f"The word {target} is not clear to me. Could you please provide me more context?"
            
        return text, option_words
            
    
    def step(self,action):
        '''
        observation, reward, terminated, truncated
        '''
        
        target = self.history[0].target
        substitutes = words = [a for a,b in self.history[0].substitutes]
        
        bot_text, option_words = self.bot_utterance(action,target)
        bot_utter = Question(bot_text,None,option_words)
        answer_reward = self.user_utterance(action, target, option_words)
        user_text = answer_reward['answer']
        user_reward = answer_reward['reward']
        user_utter = Question(user_text,None,None)

        self.history.append(bot_utter)
        self.history.append(user_utter)
        history_text = '</s>'.join([q.text for q in self.history])
        embedding = self.model.encode(history_text)

        terminated = (action == 0)
        truncated =  len(self.history)> 9
        
        reward = user_reward - 0.1  # Length penalty
        if terminated and user_reward>0:
            reward += 1
        return embedding, reward, terminated, truncated,{}
                
            
    
    def reset(self,context_id=None):
        self.history.clear()
        question = self.OA.sample(context_id)
        self.substitutes = question.substitutes
        self.target = question.target
        self.history.append(question)
        user_text = question[0]
        embedding = self.model.encode(user_text)
        # encoding and embedding
        info = {}
        return embedding, info