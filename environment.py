from collections import defaultdict,namedtuple
import gzip
import json
import random
from itertools import combinations
from transformers import AutoTokenizer, AutoModelForMaskedLM
import spacy
import pickle
import copy
import numpy as np
nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

# Question = namedtuple('question',['text','extra'])
# Question = namedtuple('question',['text','target','substitutes'])
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
        offset = target['offset']
        
        substitutes = [(sub['substitute'],sub['label_score']) for sub in target['substitutes']]
        sorted_subs = sorted(substitutes,key=lambda x:x[1],reverse=True)

        # labels = [self.swords['substitute_labels'][sid] for sid in self.tid_to_sids[target_id]]
        question = {'text':context, 'target':target_text, 'substitutes':sorted_subs,'offset':offset,'role':'user'}
        
        return question,context_id
    
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
        self.context_id = None
        self.contexts = agent.contexts
        
        self.understand_score = 0
        
        self.reward_table = {
        Action.NO_ACTION.value:{
            True:{
                'answer':'',
                'reward':2,
                'terminated':True
            },
            False:{
                'answer':" you misunderstdood my words, I mean...",
                'reward': -2,
                'terminated':True
                
        }},
        Action.CONFIRM.value:{
            True:{
                'answer':'Yes, it is',
                'reward': 1.5,
                'terminated':True
            },
            False:{
                'answer':"No, it is not ",
                'reward': -1.5,
                'terminated':False
        }},
        Action.OPTION.value:{
            True:{
                'answer':None,
                'reward':1,
                'terminated':True
            },
            False:{
                'answer':" none of these",
                'reward': -1,
                'terminated':False
        }},
        Action.EXPLAIN.value:{
            True:{
                'answer':"the explain content",
                'reward':0.5,
                'terminated':False
            },
            False:{
                'answer':"it is obviously, but I will try explain it too",
                'reward': -0.5,
                'terminated':True
        }}
        }
        with open('state.pkl','rb') as f:
            self.state_mapping = pickle.load(f)
        with open('option.pkl','rb') as f:
            self.option_mapping = pickle.load(f)
            

    
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
        should_function = [self.should_no_action, self.should_confirm,self.should_opt,self.should_explain]
        
        right_action = [f(option_words) for f in should_function].index(True)
        is_right_action = (action == right_action)
        
        answer_reward = reward_table[action][is_right_action]
        if action == Action.OPTION.value:
            answer = f'{random.choice(option_words)[0]}' if is_right_action else 'none of these'
            answer_reward['answer'] = answer

        
        # if action == Action.NO_ACTION.value:
        #     is_right_action = self.should_no_action(option_words)
        #     answer_reward = reward_table[action][is_right_action]
        # elif action == Action.CONFIRM.value:
        #     is_right_action = self.should_confirm(option_words)
        #     answer_reward = reward_table[action][is_right_action]
        # elif action == Action.OPTION.value:
        #     is_right_action = self.should_opt(option_words)
        #     answer_reward = reward_table[action][is_right_action]
        #     answer = f'{random.choice(option_words)[0]}' if is_right_action else 'none of these'
        #     answer_reward['answer'] = answer
        # else:
        #     is_right_action = self.should_explain(option_words)
        #     answer_reward = reward_table[action][is_right_action]
            
        answer_reward = copy.copy(answer_reward)
        answer_reward['is_right_action'] = is_right_action
        answer_reward['loose_right_actions'] = self.get_best_action()
        return answer_reward
        
        
    
    def get_option_words_by_llm(self,context_id,use_cache=True):
        '''
        Please do not arbitrarily alter this function.
        if so, remember updating the cache.
        '''
        
        if (context_id in self.option_mapping) and use_cache:
            return self.option_mapping[context_id]
        
        
        def repeat_part(sent,target,substitutes,trunck=False):
            rep_list = []
            substitutes = [w for w,s in substitutes[:5]]
            for sub in [target]+substitutes+['<mask>']:
                start = offset-30 if trunck else 0
                post_start = offset-sent.start_char+len(target)
                post_end = post_start+30 if trunck else  100000000
                repeat_part = f"{sent.text[start:offset-sent.start_char]}{sub}{sent.text[post_start:post_end]}"
                rep_list.append(repeat_part)
            return '.'.join(rep_list)

        h0,_ = self.OA.sample(context_id)
        # h0 = self.history[0]
        
        offset = h0['offset']
        mask_sentence_list = []
        for sent in nlp(h0['text']).sents:
            if sent.start_char <= offset <sent.end_char:
                sent_text = repeat_part(sent, h0['target'], h0['substitutes'])
                mask_sentence_list.append(sent_text)
            else:
                mask_sentence_list.append(sent.text)

        mask_text = ''.join(mask_sentence_list)
        token_lens = len(tokenizer(''.join(mask_text))['input_ids'])
        
        if token_lens>512:
            mask_sentence_list = []
            for sent in nlp(h0['text']).sents:
                if sent.start_char <= offset <sent.end_char:
                    sent_text = repeat_part(sent, h0['target'], h0['substitutes'],True)
                    # print('-------------')
                    mask_sentence_list.append(sent_text)
                else:
                    mask_sentence_list.append(sent.text)
            mask_text = ''.join(mask_sentence_list)
            token_lens = len(tokenizer(''.join(mask_text))['input_ids'])                
        words = [(token['token_str'],round(token['score'],3)) for token in self.mask_model(mask_text)]
        return words,mask_text
    
    def bot_utterance(self,action,target):
        
        option_words = self.history[0]['option_words']
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
        
        target = self.history[0]['target']
        substitutes = words = [a for a,b in self.history[0]['substitutes']]
        
        bot_text, option_words = self.bot_utterance(action,target)
        
        
        answer_reward = self.user_utterance(action, target, option_words)
        user_text = answer_reward['answer']
        user_reward = answer_reward['reward']
        
        
        
        
        terminated = answer_reward['terminated']
        truncated =  len(self.history)> 9
        reward = user_reward - 0.1  # Length penalty
        if terminated and answer_reward['is_right_action']:
            reward += 1

            
        bot_utter = {'text':bot_text, 'option_words':option_words,'action':action,'role':'bot'}
        self.history.append(bot_utter)
        user_utter = {'text':user_text,'reward':user_reward, 'is_right_action': answer_reward['is_right_action'],'loose_right_actions':answer_reward['loose_right_actions'],'role':'user'}
        self.history.append(user_utter)
        
        history_text = '</s>'.join([q['text'] for q in self.history])
        embedding = self.model.encode(history_text)
        self.history[-1]['history_text'] = history_text
        return embedding, reward, terminated, truncated,{}
    
    def get_best_action(self):
        option_words = self.history[0]['option_words']
        should_no_action = self.should_no_action(option_words)
        should_confirm = self.should_confirm(option_words)
        should_opt = self.should_opt(option_words)
        should_explain = self.should_explain(option_words)
        
        right_action = [should_no_action, should_confirm, should_opt, should_explain]
        return np.argwhere(right_action).reshape(-1)
        
            
    
    def reset(self,context_id=None):
        self.history.clear()
        info = {}

        question,context_id = self.OA.sample(context_id)
        option_words,mask_text = self.get_option_words_by_llm(context_id)
        question['option_words'] = option_words
        question['mask_text'] = mask_text

        self.context_id = context_id
        self.substitutes = question['substitutes']
        self.target = question['target']
        self.history.append(question)
        user_text = question['text']
        if context_id in self.state_mapping:
            embedding = self.state_mapping[context_id]
        else:
            embedding = self.model.encode(user_text)
        # encoding and embedding
        return embedding, info