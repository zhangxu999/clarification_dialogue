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

from rl_utils import PriorityQueue

import constant 

nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')

# Question = namedtuple('question',['text','extra'])
# Question = namedtuple('question',['text','target','substitutes'])
class Dataset:
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
    
    
    def sample(self,context_id=None,subs_num=10):
        if context_id is None:
            context_id = random.choice(self.context_ids)
        context = self.contexts[context_id]['context']
        
        target = self.contexts[context_id]['targets'][0]
        target_text = target['target']
        lemma_target = lemmatizer.lemmatize(target_text)
        
        offset = target['offset']
        
        substitutes = [(sub['substitute'],sub['label_score']) for sub in target['substitutes']]
        sorted_subs = sorted(substitutes,key=lambda x:x[1],reverse=True)
        sorted_subs = [(w,s) for w,s in sorted_subs if len(w.split())==1][:subs_num]
        lemma_subs = [(lemmatizer.lemmatize(w),s) for (w,s) in sorted_subs]

        # labels = [self.swords['substitute_labels'][sid] for sid in self.tid_to_sids[target_id]]
        question = {'text':context, 'target':target_text, 'lemma_target':lemma_target
                    ,'substitutes':sorted_subs,'lemma_subs':lemma_subs,'offset':offset,'role':'user'}
        
        return question,context_id
    
    def answer(self):
        context = self.sample()[0]
        return context[:20]
        
        
from constant import Action

class User:
    
    def __init__(self,data_agent):
        self.data_agent = data_agent
        self.contexts = data_agent.contexts
        self.context_ids = list(self.contexts.keys())
        self.target = None
        self.lemma_subs = []
        self.best_word = (None,0)
        
        self.find_subs = False
    
    def init_dialoag(self,context_id=None,subs_num=10):
        self.find_subs = False
        
        if context_id is None:
            context_id = random.choice(self.context_ids)
        context = self.contexts[context_id]['context']
        self.target = target = self.contexts[context_id]['targets'][0]
        target_text = target['target']
        self.lemma_target = lemma_target = lemmatizer.lemmatize(target_text)
        offset = target['offset']
        substitutes = [(sub['substitute'],sub['label_score']) for sub in target['substitutes']]
        sorted_subs = sorted(substitutes,key=lambda x:x[1],reverse=True)
        sorted_subs = [(w,s) for w,s in sorted_subs if len(w.split())==1][:subs_num]
        self.lemma_subs = lemma_subs = [(lemmatizer.lemmatize(w),s) for (w,s) in sorted_subs]
        # labels = [self.swords['substitute_labels'][sid] for sid in self.tid_to_sids[target_sub_maps = }
        self.lemma_sub_maps = {k:v for k,v in self.lemma_subs}
        self.highscore_subs = {k:v for k,v in self.lemma_subs[:5]}
        
        self.highscore_subs[self.lemma_target] = 1.5
        question = {'text':context, 'target':target_text, 'lemma_target':lemma_target
                    ,'substitutes':sorted_subs,'lemma_subs':lemma_subs,'offset':offset,'role':'user'}
        return question, context_id
    
    def is_find_subs(self):
        return self.find_subs
    
    
        
    def is_right_action(self, action, option_words):
        
        '''
        terminated 条件：
        1. 找到了target，或high substitutes,
        2. options words 用完了
        3. 长度超过限度
        '''

        
        if len(option_words)==0:
            is_right = False
            
        elif action ==Action.NO_ACTION.value:
            is_right = option_words[0] == self.lemma_target
            if is_right:
                self.find_subs = True
        elif action in (Action.CONFIRM.value, Action.OPTION.value):
            is_right = any([w in self.highscore_subs for w in option_words])
            if is_right:
                self.find_subs = True
        else:
            is_right = all([w not in self.highscore_subs for w in option_words])
        
        
        reward_list = [3, 2, 1, 0]
        reward = reward_list[action]

        if not is_right:
            reward *=-1
            
            
        terminated_table = [[True, True],
                            [False, True],
                            [False, True],
                            [False, False]]
        if len(option_words)==0:
            terminated = True
        else:
            terminated = terminated_table[action][int(is_right)]
        
        answer_table = [[" you misunderstdood my words, I mean...",''],
          ['No, it is not','Yes, it is'],
          ["none of these",f"the first"],
          ["it is obviously, but I will try explain it too","the explain content"]]
        answer = answer_table[action][int(is_right)]
        
        if action == Action.OPTION.value:
            words = [w for w in option_words if w in self.highscore_subs]
            answer = f'{",".join(words)}' if is_right else 'none of these'
        
        return is_right, reward, terminated, answer, self.find_subs

    def utterance(self,action, option_words):
        # should_function = [self.should_no_action, self.should_confirm, self.should_opt, self.should_explain]
        
        is_right_action,reward,terminated, answer,find_subs = self.is_right_action(action,option_words)
        
        
        answer_reward = dict(reward=reward,terminated=terminated,text=answer, is_right_action=is_right_action,find_subs=find_subs)
        # answer_reward['reward'] = reward
        # answer_reward['is_right_action'] = is_right_action
        
        return answer_reward
    
    
    
class Agent:
    
    def __init__(self):
        self.asked_words = []
        self.option_words = None
      
    def push_option_words(self,option_words):
        self.option_words = PriorityQueue()
        for w, s in option_words:
            self.option_words.push((w,s),-1*s)
        # print('-----------------------------------')
        # print([c[0] for a,b,c in self.option_words._queue])
        
    def utterance(self,action,target):
        # print(action, [c[0] for a,b,c in self.option_words._queue])
        words = []
        if action == Action.NO_ACTION.value:
            text = f"I'm pretty sure of the meaning of the word {target} . "
            if not self.option_words.is_empty():
                word,score = self.option_words.pop()
                words.append(word)
        elif action == Action.CONFIRM.value:

            if not self.option_words.is_empty():
                word,score = self.option_words.pop()
                self.asked_words.append(word)
                words.append(word)
            if len(words)>0:
                text = f"The word {target} is not clear to me. Do you mean something like {','.join(words)} by this?"
            else:
                text = f" there is no enough words to discuss."
        elif action == Action.OPTION.value:
            for i in range(3):
                if not self.option_words.is_empty():
                    word,score = self.option_words.pop()
                    self.asked_words.append(word)
                    words.append(word)
            if len(words)>0:
                text = f"The word {target} is not clear to me. Do you mean something like {','.join(words)} by this?"
            else:
                text = f" there is no enough words to discuss."
        elif action == Action.EXPLAIN.value:

            text = f"The word {target} is not clear to me. Could you please provide me more context?"
            if not self.option_words.is_empty():
                word,score = self.option_words.pop()
                words.append(word)

        bot_response = dict(action=action,text=text,option_words=words)
        return bot_response
        
        

class DialougeEnv:
    
    def __init__(self,dataset, model, mask_model,device, length_penalty=0.5,max_length=9,Debug=False):
        
        self.device = device
        
        self.length_penalty = length_penalty
        self.max_length = max_length
        
        self.Debug = Debug
        
        self.dataset = dataset
        self.user = User(dataset)
        self.agent = Agent()
        
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
        
        
        self.model = model
        self.mask_model = mask_model
        
        self.history = []
        self.substitutes = None
        self.target = None
        self.context_id = None
        self.contexts = dataset.contexts
        
        with open('data/state.pkl','rb') as f:
            self.state_mapping = pickle.load(f)
        with open('data/option.pkl','rb') as f:
            self.option_mapping = pickle.load(f)
                    
    
    def get_option_words_by_llm(self,context_id,use_cache=True, options_num=10):
        '''
        Please do not arbitrarily alter this function.
        if so, remember updating the cache.
        '''

        if (context_id in self.option_mapping) and use_cache:
            return self.option_mapping[context_id]
        
        
        def repeat_part(sent,target,substitutes,trunck=False):
            rep_list = []
            substitutes = [w for w,s in substitutes if len(w.split())==1]
            for sub in [target]+substitutes+['<mask>']:
                start = offset-30 if trunck else 0
                post_start = offset-sent.start_char+len(target)
                post_end = post_start+30 if trunck else 100000000
                repeat_part = f"{sent.text[start:offset-sent.start_char]}{sub}{sent.text[post_start:post_end]}"
                rep_list.append(repeat_part)
            return ' '.join(rep_list)

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

        mask_text = ' '.join(mask_sentence_list)
        token_lens = len(tokenizer(' '.join(mask_text))['input_ids'])
        
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
        origin_words = [(token['token_str'],round(token['score'],3)) for token in self.mask_model(mask_text,top_k=20)]
        words = [(w,s) for w,s in origin_words if w not in constant.filter_words]
        lemmatized_words = [(lemmatizer.lemmatize(w),s) for w,s in words[:options_num]]
        # print(origin_words,'-----------------\n',words,'-----------------\n',lemmatized_words)
        return lemmatized_words, mask_text
        
    

    def step(self,action):
        '''
        observation, reward, terminated, truncated
        '''
        target = self.history[0]['target']
        agent_response = self.agent.utterance(action,target)
        agent_response['role'] = 'bot'
        self.history.append(agent_response)


        action = agent_response['action']
        option_words = agent_response['option_words']
        user_response = self.user.utterance(action, option_words)
        user_response['role'] = 'user'

        truncated =  len(self.history) > self.max_length
        terminated = (user_response['terminated'] or truncated)
        reward_detail = {'action_reward':user_response['reward'],
                         'length_penalty':self.length_penalty
                        }
        reward = user_response['reward'] - self.length_penalty  # Length penalty
        if terminated and self.user.is_find_subs():
            reward_detail['find_subs'] = 3
            reward += 3
        user_response['terminated'] = terminated
        user_response['reward'] = reward
        user_response['reward_detail']  = reward_detail
        self.history.append(user_response)
        if self.Debug:
            print("-------------------")
            print(agent_response)
            print(user_response)


        history_text = '</s>'.join([q['text'] for q in self.history])
        topn_words = self.agent.option_words.topn(3)
        
        history_text = f"{history_text}</s>{''.join(topn_words)}"
        
        embedding = self.model.encode(history_text)
        # self.history[-1]['history_text'] = history_text
        return embedding, reward, terminated, truncated, {}
            
            
    
    def reset(self,context_id=None):
        self.history.clear()
        info = {}
        
        question,context_id = self.user.init_dialoag(context_id,subs_num=15)
        option_words,mask_text = self.get_option_words_by_llm(context_id,use_cache=True,options_num=5)
        
        self.agent.push_option_words(option_words)
        question['option_words'] = option_words
        question['mask_text'] = mask_text
        # print(mask_text)
        # print(option_words)
        
        
        self.context_id = context_id
        self.substitutes = question['substitutes']
        self.lemma_subs = question['lemma_subs']
        self.target = question['target']
        self.lemma_target = question['lemma_target']
        self.history.append(question)
        user_text = question['text']
        
        embedding_text = f"{user_text}</s>{option_words[:3]}"
        embedding = self.model.encode(embedding_text)
        
        # if context_id in self.state_mapping:
        #     embedding = self.state_mapping[context_id]
        # else:
        #     embedding = self.model.encode(embedding_text)
        # encoding and embedding
        if self.Debug:
            print("#################################################################")
            print(self.target,self.lemma_target)
            print(self.lemma_subs[:10])
            print(option_words)
        return embedding, info