from collections import defaultdict,namedtuple
import gzip
import json
import random

Question = namedtuple('question',['context','target','substitutes'])
class OriginAgent:
    def __init__(self):
        with gzip.open('../data/swords/swords-v1.1_dev.json.gz', 'r') as f:
            self.swords = json.load(f)

        # Gather substitutes by target
        self.tid_to_sids = defaultdict(list)
        for sid, substitute in self.swords['substitutes'].items():
             self.tid_to_sids[substitute['target_id']].append(sid)

        # i =0
        # # Iterate through targets
        self.target_ids = list(self.swords['targets'].keys())
        # for tid, target in swords['targets'].items():
        #   context = swords['contexts'][target['context_id']]
        #   substitutes = [swords['substitutes'][sid] for sid in tid_to_sids[tid]]
        #   labels = [swords['substitute_labels'][sid] for sid in tid_to_sids[tid]]
        #   scores = [l.count('TRUE') / len(l) for l in labels]
        #   print('-' * 80)
        #   print(context['context'])
        #   print('-' * 20)
        #   print('{} ({})'.format(target['target'], target['pos']))
        #   print(', '.join(['{} ({}%)'.format(substitute['substitute'], round(score * 100))\
        #                    for substitute, score in sorted(zip(substitutes, scores), key=lambda x: -x[1])]))
        
    def sample(self):
        target_id = random.choice(self.target_ids)
        target = self.swords['targets'][target_id]
        context = self.swords['contexts'][target['context_id']]
        substitutes = [self.swords['substitutes'][sid]['substitute'] for sid in self.tid_to_sids[target_id]]
        # labels = [self.swords['substitute_labels'][sid] for sid in self.tid_to_sids[target_id]]
        
        return Question(context['context'],target['target'],substitutes)
    
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
    
    def step(self,action):
        '''
        observation, reward, terminated, truncated
        '''
        
        if action == 0:
            answer = self.OA.answer()
            self.history.clear()
            return None, 0, True, False,{}
        else:
            idx = -1
            while self.history[idx].target is None:
                idx -=1
            target = self.history[idx].target
            words = self.history[idx].substitutes
            if action == 1:
                response = f"The word {target} is not clear to me. Do you mean something like {words[0]} by this?"
                bot_utter = Question(response,None,None)
                new_user_utter = Question("Yes, it is",None,None)
            elif action== 2:
                response = f"The word {target} is not clear to me. Do you mean something like {words[0]},{words[1]},{words[2]} by this?"
                bot_utter = Question(response,None,None)
                new_user_utter = Question(f"like {words[1]}",None,None)
            elif action == 3:
                response = f"The word {target} is not clear to me. Could you please provide me more context?"
                bot_utter = Question(response,None,None)
                new_user_utter = Question("Sure, I would like to explain it",None,None)
            self.history.append(bot_utter)
            self.history.append(new_user_utter)
            history_text = '</s>'.join([q.context for q in self.history])
            embedding = self.model.encode(history_text)
            reward = random.random() - 0.03*len(self.history)
            terminated = False
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