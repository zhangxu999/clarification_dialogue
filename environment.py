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

nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

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
        
        
from enum import Enum
from constant import reward_table
# 定义一个枚举类型
class Action(Enum):
    NO_ACTION = 0
    CONFIRM = 1
    OPTION = 2
    EXPLAIN = 3

class User:
    
    def __init__(self,data_agent):
        self.data_agent = data_agent
        self.contexts = data_agent.contexts
        self.context_ids = list(self.contexts.keys())
        self.reward_table = reward_table
        self.target = None
        self.lemma_subs = []
        self.best_word = (None,0)
    
    def init_dialoag(self,context_id=None,subs_num=10):
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
        self.highscore_subs = {k:v for k,v in self.lemma_subs if v >= 0.7}
        question = {'text':context, 'target':target_text, 'lemma_target':lemma_target
                    ,'substitutes':sorted_subs,'lemma_subs':lemma_subs,'offset':offset,'role':'user'}
        return question, context_id
        
    def is_right_action(self, action, option_words):
        if action ==Action.NO_ACTION.value:
            is_right = option_words[0] == self.target
        elif action in (Action.CONFIRM.value, Action.OPTION.value):
            is_right = any([w in self.highscore_subs for w in option_words])
        else:
            is_right = all([w not in self.highscore_subs for w in option_words])
        
        reward_list = [3, 2, 1, 0.5]
        reward = reward_list[Action]
        if not is_right:
            reward *=-1
        
        return is_right, reward

    def utterance(self,action, option_words):
        reward_table = self.reward_table
        # should_function = [self.should_no_action, self.should_confirm,self.should_opt,self.should_explain]
        
        is_right_action,reward = self.is_right_action(action,option_words)
        
        answer_reward = reward_table[action][is_right_action]
        if action == Action.OPTION.value:
            words == [w for w in option_words if w in self.highscore_subs]
            answer = f'{",".join(words)}' if is_right_action else 'none of these'
            answer_reward['answer'] = answer
        
        answer_reward = copy.copy(answer_reward)
        answer_reward['reward'] = reward
        answer_reward['is_right_action'] = is_right_action
        
        # answer_reward['loose_right_actions'] = self.get_best_action()
        return answer_reward
    
    
    
class Agent:
    
    def __init__(self):
        self.asked_words = []
        self.option_words = PriorityQueue()
        
    def push_option_words(self,option_words):
        for w, s in option_words:
            self.option_words.push(w,-1*s)
        
    def utterance(self,action,target):
        words = []
        option_words = self.history[0]['option_words']
        if action == Action.NO_ACTION.value:
            text = f"I'm pretty sure of the meaning of the word {target} . "
            word,score = self.option_words.nsmallest(1)
            words.append(word)
        elif action == Action.CONFIRM.value:
            if not self.option_words.is_empty():
                word,score = self.option_words.pop()
                self.asked_words.append(word)
                words.append(word)
            text = f"The word {target} is not clear to me. Do you mean something like {word} by this?"
        elif action == Action.OPTION.value:
            for i in range(3):
                if not self.option_words.is_empty():
                    word,score = self.option_words.pop()
                    self.asked_words.append(word)
                    words.append(word)
            text = f"The word {target} is not clear to me. Do you mean something like {','.join(words)} ?"
        elif action == Action.EXPLAIN.value:
            text = f"The word {target} is not clear to me. Could you please provide me more context?"
            word,score = self.option_words.smallest(1)
            words.append(word)

        bot_response = dict(action=action,text=text,option_words=words)
        return bot_response
        
        

class DialougeEnv:
    
    def __init__(self,agent,model,mask_model, user, agent):
        
        self.user = user
        self.agent = agent
        
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
        
        with open('state.pkl','rb') as f:
            self.state_mapping = pickle.load(f)
        with open('option.pkl','rb') as f:
            self.option_mapping = pickle.load(f)
            

    def user_utterance(self,action,target,option_words):
        
        reward_table = self.reward_table
        should_function = [self.should_no_action, self.should_confirm, self.should_opt, self.should_explain]
        right_action = [f(option_words) for f in should_function].index(True)
        is_right_action = (action == right_action)
        
        answer_reward = reward_table[action][is_right_action]
        if action == Action.OPTION.value:
            answer = f'{random.choice(option_words)[0]}' if is_right_action else 'none of these'
            answer_reward['answer'] = answer

            
        answer_reward = copy.copy(answer_reward)
        answer_reward['is_right_action'] = is_right_action
        answer_reward['loose_right_actions'] = self.get_best_action()
        return answer_reward
        
    
    def get_option_words_by_llm(self,context_id,use_cache=True, options_num=10):
        '''
        Please do not arbitrarily alter this function.
        if so, remember updating the cache.
        '''

        if (context_id in self.option_mapping) and use_cache:
            return self.option_mapping[context_id]

        
        filter_words = '''</s>
        .
        ..
        ?
        s
        ''
        !”
        .”
        ”
        nos
        mr
        ve
        '''.split()+ ['']
        
        
        
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
        words = [(w,s) for w,s in origin_words if w not in filter_words]
        lemmatized_words = [(lemmatizer.lemmatize(w),s) for w,s in words[:options_num]]
        # print(origin_words,'-----------------\n',words,'-----------------\n',lemmatized_words)
        return lemmatized_words, mask_text
    
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
        agent_response = self.agent.utterance(action,target)
        
        action = agent_response['action']
        user_response = self.user.utterance(action, target, option_words)
        user_text = user_response['answer']
        user_reward = user_response['reward']
        
        
        terminated = answer_reward['terminated']
        truncated =  len(self.history) > 9
        reward = user_reward - 0.1  # Length penalty
        if terminated and answer_reward['is_right_action']:
            reward += 3
            
        bot_utter = {'text':bot_text, 
                     'option_words':option_words,
                     'action':action,
                     'role':'bot'
                    }
        self.history.append(bot_utter)
        user_utter = {'text':user_text,
                      'reward':user_reward,
                      'is_right_action': answer_reward['is_right_action'],
                      'role':'user'}
        self.history.append(user_utter)
        
        history_text = '</s>'.join([q['text'] for q in self.history])
        embedding = self.model.encode(history_text)
        self.history[-1]['history_text'] = history_text
        return embedding, reward, terminated, truncated, {}
            
            
    
    def reset(self,context_id=None):
        self.history.clear()
        info = {}

        question,context_id = self.OA.sample(context_id,subs_num=5)
        option_words,mask_text = self.get_option_words_by_llm(context_id,use_cache=True,options_num=5)
        
        self.user.init_dialoag()
        
        self.agent.push_option_words(option_words)
        question['option_words'] = option_words
        question['mask_text'] = mask_text
        
        
        self.context_id = context_id
        self.substitutes = question['substitutes']
        self.lemma_subs = question['lemma_subs']
        self.target = question['target']
        self.lemma_target = question['lemma_target']
        self.history.append(question)
        user_text = question['text']
        if context_id in self.state_mapping:
            embedding = self.state_mapping[context_id]
        else:
            embedding = self.model.encode(user_text)
        # encoding and embedding
        return embedding, info