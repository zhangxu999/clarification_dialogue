from collections import defaultdict,namedtuple
import gzip
import json
import random
from itertools import combinations

Question = namedtuple('question',['text','target','substitutes'])
class OriginAgent:
    def __init__(self):
        with gzip.open('../data/swords/swords-v1.1_dev.json.gz', 'r') as f:
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
    
    
    def sample(self):
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
        
        

class DialougeEnv:
    
    def __init__(self,agent,model):
        
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
        
        self.history = []
        self.contexts = agent.contexts
        
        self.understand_score = 0
        
    def should_confirm(self,subs,sub):
        is_top = (sub == subs[0][0])  # the top one  have a high score.
        is_have_gap = (subs[0][1] - subs[1][1])>0.3 # the score of top subtitute is much larger than the second one.
        return is_top and is_have_gap
    
    def should_opt(self,sorted_subs,sub_words):
        if (sub_words[0] not in [sorted_subs[i][0] for i in range(3)]):
            return False

        for (a_word,a_score), (b_word, b_score) in combinations(sorted_subs[:3],2):
            if round(abs(a_score-b_score),1) > 0.2:
                return False
        return True
        
    def user_utterance(self,action,target,option_words):
        substitutes = self.history[0].substitutes
        
        if action == 0:
            # when the action is "NO Action
            if len(substitutes) < 10:
                answer = ''
                reward = 1
                self.understand_score = 0.9
            else:
                answer = ''
                reward = -1
                self.understand_score = 0.1
        elif action == 1:
            # action is "Confirm"
            confirm_word = option_words[0]
            if self.should_confirm(substitutes, confirm_word):
                answer = 'Yes, it is'
                reward = 1
                self.understand_score += 0.4
            else:
                answer = 'No, it is not'
                reward = 0
                self.understand_score += 0.1
        elif action == 2:
            # options
            if self.should_opt(substitutes,option_words):
                answer = f"{random.choice(option_words)}"
                reward = 2
                self.understand_score += 0.6
            else:
                answer = 'None of these words'
                reward = 0
                self.understand_score += 0.2
        else:
            # more information
            answer = "the explain content"
            reward = 0
            self.understand_score += 0.3
        return answer, reward, self.understand_score
    
    def bot_utterance(self,action,target):
        
        substitutes = self.history[0].substitutes
        sub_words = [a for a,b, in substitutes]
        scores =  [b for a,b, in substitutes]
        words = None
        word_index = random.choices(range(len(sub_words[:-3])), weights=scores[:-3])[0]  # sample words by score distribution
        if action == 0:
            text = ''
        elif action == 1:
            words = [sub_words[word_index]]
            text = f"The word {target} is not clear to me. Do you mean something like {words[0]} by this?"
            
        elif action == 2:
            words = sub_words[word_index:word_index+3]
            text = f"The word {target} is not clear to me. Do you mean something like {words[0]},{words[1]},{words[2]} by this?"
        else:
            text = f"The word {target} is not clear to me. Could you please provide me more context?"
            
        return text,words
            
    
    def step(self,action):
        '''
        observation, reward, terminated, truncated
        '''
        
        target = self.history[0].target
        substitutes = words = [a for a,b in self.history[0].substitutes]
        
        bot_text, words = self.bot_utterance(action,target)
        bot_utter = Question(bot_text,None,None)
        user_text, reward, understand_score = self.user_utterance(action, target, words)
        user_utter = Question(user_text,None,None)

        self.history.append(bot_utter)
        self.history.append(user_utter)
        history_text = '</s>'.join([q.text for q in self.history])
        embedding = self.model.encode(history_text)

        terminated = (action == 0) or (understand_score>1)
        truncated =  len(self.history)> 8
        return embedding, reward, terminated, truncated,{}
                
            
    
    def reset(self):
        self.history.clear()
        question = self.OA.sample()
        self.history.append(question)
        user_text = question[0]
        embedding = self.model.encode(user_text)
        # encoding and embedding
        info = {}
        return embedding, info