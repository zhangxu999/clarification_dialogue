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
from functools import lru_cache

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
    
    def __init__(self,data_agent,reward_func='is_right_action'):
        self.data_agent = data_agent
        self.contexts = data_agent.contexts
        self.context_ids = list(self.contexts.keys())
        self.target = None
        self.lemma_subs = []
        self.best_word = (None,0)
        
        self.reward_func = reward_func
        self.find_subs = False
        
        self.explained = False
    
    def init_dialoag(self,context_id=None,subs_num=10):
        self.find_subs = False
        self.explained = False
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
    
    
        
    def is_right_action_score(self, action, option_words):
        '''
        terminated 条件：
        1. 找到了target，或high substitutes,
        2. options words 用完了
        3. 长度超过限度 
        '''
        len_option = len(option_words)
        if len_option == 1:
            gap = 0
        if len_option == 2:
            gap = option_words[0][1] - option_words[1][1]
        elif len_option == 3:
            gap1 = option_words[0][1] - option_words[1][1]
            gap2 = option_words[0][1] - option_words[2][1]
            gap = max(gap1, gap2)
            
        
        scores = [s for w,s in option_words]
        
        if len(option_words)==0:
            is_right = False
        
        elif action == Action.NO_ACTION.value:
            is_right = (option_words[0][1] >=0.6) or (gap >= 0.3)
        elif action ==Action.CONFIRM.value:
            is_right = (gap >= 0.2)
        elif action == Action.OPTION.value:
            is_right =   (gap >=0.1) and max(scores)> 0
        else:
            is_right = max(scores) == 0
            # is_right = all([w not in self.highscore_subs for w in option_words])
        
        self.find_subs = any([w in self.highscore_subs for w,s in option_words])
        
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
        elif (action == Action.EXPLAIN.value and self.explained):
            terminated = True
        else:
            terminated = terminated_table[action][int(is_right)]
        
        answer_table = [[" you misunderstdood my words, I mean...",''],
          ['No, it is not','Yes, it is'],
          ["none of these",f"the first"],
          ["it is obviously, but I will try explain it too","sure"]]
        answer = answer_table[action][int(is_right)]
        
        if action == Action.OPTION.value:
            words = [w for w in option_words if w in self.highscore_subs]
            answer = f'{",".join(words)}' if is_right else 'none of these'

        
        return is_right, reward, terminated, answer, self.find_subs

    def is_right_action(self, action, option_words):
        
        '''
        terminated 条件：
        1. 找到了target，或high substitutes,
        2. options words 用完了
        3. 长度超过限度 
        '''
        words = [w for w,s in option_words]
        scores = [s for w,s in option_words]

        
        if len(option_words)==0:
            is_right = False
            
        elif action ==Action.NO_ACTION.value:
            is_right = words[0] == self.lemma_target
            if is_right:
                self.find_subs = True
        elif action in (Action.CONFIRM.value, Action.OPTION.value):
            is_right = any([w in self.highscore_subs for w in words])
            if is_right:
                self.find_subs = True
        else:
            is_right = all([w not in self.highscore_subs for w in words])
        
        
        reward_list = [3, 2, 1, 0]
        reward = reward_list[action]

        if not is_right:
            reward *=-1
            
        
        terminated_table = [[True, True],
                            [False, True],
                            [False, True],
                            [False, False]]
        if len(words)==0:
            terminated = True
        else:
            terminated = terminated_table[action][int(is_right)]
        
        answer_table = [[" you misunderstdood my words, I mean...",''],
          ['No, it is not','Yes, it is'],
          ["none of these",f"the first"],
          ["it is obviously, but I will try explain it too","sure"]]
        answer = answer_table[action][int(is_right)]
        
        if action == Action.OPTION.value:
            words = [w for w in words if w in self.highscore_subs]
            answer = f'{",".join(words)}' if is_right else 'none of these'
            
        
        return is_right, reward, terminated, answer, self.find_subs

    def utterance(self,action, option_words):
        # should_function = [self.should_no_action, self.should_confirm, self.should_opt, self.should_explain]
        reward_func = getattr(self, self.reward_func)
        is_right_action,reward,terminated, answer,find_subs = reward_func(action,option_words)
        
        
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
            if w not in self.asked_words:
                self.option_words.push((w,s),-1*s)
        # print('-----------------------------------')
        # print([c[0] for a,b,c in self.option_words._queue])
        
    def utterance(self,action,target):
        # print(action, [c[0] for a,b,c in self.option_words._queue])
        words = []
        word_scores = []
        if action == Action.NO_ACTION.value:
            text = f"I'm pretty sure of the meaning of the word {target} . "
            if not self.option_words.is_empty():
                word,score = self.option_words.pop()
                words.append(word)
                word_scores.append((word,score))
        elif action == Action.CONFIRM.value:

            if not self.option_words.is_empty():
                word,score = self.option_words.pop()
                words.append(word)
                word_scores.append((word,score))
            if len(words)>0:
                text = f"The word {target} is not clear to me. Do you mean something like {','.join(words)} by this?"
            else:
                text = f" there is no enough words to discuss."
        elif action == Action.OPTION.value:
            for i in range(3):
                if not self.option_words.is_empty():
                    word,score = self.option_words.pop()
                    words.append(word)
                    word_scores.append((word,score))
            if len(words)>0:
                text = f"The word {target} is not clear to me. Do you mean something like {','.join(words)} by this?"
            else:
                text = f" there is no enough words to discuss."
        elif action == Action.EXPLAIN.value:

            text = f"The word {target} is not clear to me. Could you please provide me more context?"
            if not self.option_words.is_empty():
                word,score = self.option_words.pop()
                words.append(word)
                word_scores.append((word,score))
                
        self.asked_words += words
                

        bot_response = dict(action=action,text=text,option_words=word_scores)
        return bot_response
        

        
        

class DialougeEnv:
    
    def __init__(self,dataset, model, mask_model,device, 
                 length_penalty=0.5,max_length=9,debug=False,append_words=True,
                 append_score=False,addscore_emb=False,reward_func='is_right_action',
                w1=1/3,w2=1/3,w3=1/3):
        
        
        
        self.device = device
        
        self.length_penalty = length_penalty
        self.max_length = max_length
        
        self.debug = debug
        self.append_words = append_words
        self.append_score = append_score
        self.addscore_emb = addscore_emb
        
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        
        self.dataset = dataset
        self.user = User(dataset,reward_func)
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
        self.context_info = None
        
        with open('data/state.pkl','rb') as f:
            self.state_mapping = pickle.load(f)
        with open('data/option.pkl','rb') as f:
            self.option_mapping = pickle.load(f)
            
    def set_debug(self,debug):
        self.debug=debug
        
    # @lru_cache(10000)                
    def generate_mask_text(self, text, offset, target, substitutes, more_context=False):
        def split_doc(text, offset):
            pre_list = []
            post_list = []
            target_sentence = None
            is_pre_sent = True
            pre_tokens = []
            post_tokens = []
            target_token = None
            is_pre_token =True

            for sent in nlp(text).sents:
                if sent.start_char <= offset < sent.end_char:
                    target_sentence = sent
                    sen_offset = offset - target_sentence.start_char

                    is_pre_sent = False
                    for token in target_sentence:
                        if token.idx-target_sentence.start_char==sen_offset:
                            target_token = token
                            is_pre_token = False
                        else:
                            if is_pre_token:
                                pre_tokens.append(token)
                            else:
                                post_tokens.append(token)
                    # sent_text = repeat_part(sent, h0['target'], h0['substitutes'][:5])
                    # mask_sentence_list.append(sent_text)
                else:
                    if is_pre_sent:
                        pre_list.append(sent)
                    else:
                        post_list.append(sent)
            return pre_list,target_sentence, post_list, pre_tokens, target_token, post_tokens

        pre_list,target_sentence, post_list, pre_tokens, target_token, post_tokens = split_doc(text,offset)

        extend = 0
        while True:
            items = []
            subs = [w for w,s in substitutes]
            for sub in [target,*subs, '<mask>']:
                pre_part = pre_tokens[extend:]
                post_part = post_tokens[:len(post_tokens) - extend]
                pre_part = ' '.join([t.text for t in pre_part])
                post_part = ' '.join([t.text for t in post_part])
                one_item = f"{pre_part} {sub} {post_part}".strip()
                if (not one_item.endswith(',')) and (not one_item.endswith('.')):
                    one_item += ','
                items.append(one_item)
            mask_text = ' '.join(items)
            if more_context:
                pre_sent = ''.join([sent.text for sent in pre_list])
                post_sent = ''.join([sent.text for sent in post_list])
                mask_text = f"{pre_sent} {mask_text} {post_sent}"
            token_lens = len(tokenizer(mask_text)['input_ids'])
            extend += 1
            if token_lens <=512:
                break
        return mask_text

    
    @lru_cache(10000)
    def get_mask_words(self,mask_text, top_k):
        origin_words = [(token['token_str'],round(token['score'],3)) for token in self.mask_model(mask_text,top_k=top_k)]
        return origin_words
    
    
    def get_option_words_by_llm(self,question,use_cache=True, options_num=10,more_context=False):
        '''
        Please do not arbitrarily alter this function.
        if so, remember updating the cache.
        '''

        # if (context_id in self.option_mapping) and use_cache:
        #     return self.option_mapping[context_id]
        
        text, offset, target, substitutes = question['text'], question['offset'], question['target'], question['substitutes'][:5]
        mask_text = self.generate_mask_text(text, offset, target, substitutes,more_context)
        
        origin_words = self.mask_model.get(mask_text, top_k=20)
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
        
        if action == Action.EXPLAIN.value:
            new_option_words, mask_text = self.get_option_words_by_llm(self.context_info, more_context=True)
            self.agent.push_option_words(new_option_words)
            self.context_info['option_words'] = option_words
            self.context_info['mask_text'] = mask_text

            

        truncated =  len(self.history) > self.max_length 
        terminated = (user_response['terminated'] or truncated)
        
        R_match = user_response['reward'] #- self.length_penalty  # Length penalty
        R_find = 3 if (terminated and self.user.is_find_subs()) else 0
        R_length = 1/self.length_penalty if terminated else 0
        
        R = self.w1 * R_match + self.w2 * R_find + self.w3 * R_length
        user_response['terminated'] = terminated
        user_response['reward'] = R
        
        reward_detail = {'action_reward': self.w1 * R_match,
                         'find_reward': R_find,
                         'length_penalty':self.w3 * R_length
                        }
        user_response['reward_detail']  = reward_detail
        self.history.append(user_response)
        if self.debug:
            print("-------------------")
            print(agent_response)
            print(user_response)

        
        history_text = '</s>'.join([q['text'] for q in self.history])
        
        topn_words_score = self.agent.option_words.topn(3,with_score=True)
        if self.append_words:
            topn_words = [w for w,s in topn_words_score]
            history_text = f"{history_text}</s>{','.join(topn_words)}"
        if self.append_score:
            w_s_text = [f"{w},{int(s)}" for w,s in topn_words_score]
            history_text = f"{history_text}</s>{';'.join(w_s_text)}"
            

        embedding = self.model.encode(history_text)
            
        
        if self.addscore_emb:
            scores = np.array([0,0,0])
            w_scores = [s for w,s in topn_words_score]
            for i,s in enumerate(w_scores):
                scores[i] = s
            embedding = np.concatenate([embedding, scores])
            
            
    
        # self.history[-1]['history_text'] = history_text
        return embedding, R, terminated, truncated, {}
            
            
    
    def reset(self,context_id=None):
        self.history.clear()
        info = {}
        
        context_info,context_id = self.user.init_dialoag(context_id,subs_num=15)
        
        option_words,mask_text = self.get_option_words_by_llm(context_info, use_cache=True,options_num=5)
        
        
        self.agent.asked_words.clear()
        self.agent.push_option_words(option_words)
        context_info['option_words'] = option_words
        context_info['mask_text'] = mask_text
        # print(mask_text)
        # print(option_words)
        
        self.context_info = context_info
        self.context_id = context_id
        self.substitutes = context_info['substitutes']
        self.lemma_subs = context_info['lemma_subs']
        self.target = context_info['target']
        self.lemma_target = context_info['lemma_target']
        self.history.append(context_info)
        history_text = context_info['text']
        
        
        topn_words_score = self.agent.option_words.topn(3,with_score=True)
        if self.append_words:
            topn_words = [w for w,s in topn_words_score]
            history_text = f"{history_text}</s>{','.join(topn_words)}"
        if self.append_score:
            w_s_text = [f"{w},{int(s)}" for w,s in topn_words_score]
            history_text = f"{history_text}</s>{';'.join(w_s_text)}"
            
        embedding = self.model.encode(history_text)
        
        
        if self.addscore_emb:
            scores = np.array([0,0,0])
            w_scores = [s for w,s in topn_words_score]
            for i,s in enumerate(w_scores):
                scores[i] = s
            embedding = np.concatenate([embedding, scores])

        # if context_id in self.state_mapping:
        #     embedding = self.state_mapping[context_id]
        # else:
        #     embedding = self.model.encode(embedding_text)
        # encoding and embedding
        if self.debug:
            print("#################################################################")
            print(self.target,self.lemma_target)
            print(self.lemma_subs[:10])
            print(option_words)
        return embedding, info