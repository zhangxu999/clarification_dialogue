# 定义一个枚举类型
from enum import Enum

class Action(Enum):
    NO_ACTION = 0
    CONFIRM = 1
    OPTION = 2
    EXPLAIN = 3

    
filter_words = '''</s>
        .
        ...
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

# reward_table = {
#         Action.NO_ACTION.value:{
#             True:{
#                 'answer':'',
#                 'reward':2,
#                 'terminated':True
#             },
#             False:{
#                 'answer':" you misunderstdood my words, I mean...",
#                 'reward': -2,
#                 'terminated':True
                
#         }},
#         Action.CONFIRM.value:{
#             True:{
#                 'answer':'Yes, it is',
#                 'reward': 1.5,
#                 'terminated':True
#             },
#             False:{
#                 'answer':"No, it is not ",
#                 'reward': -1.5,
#                 'terminated':False
#         }},
#         Action.OPTION.value:{
#             True:{
#                 'answer':None,
#                 'reward':1,
#                 'terminated':True
#             },
#             False:{
#                 'answer':"none of these",
#                 'reward': -1,
#                 'terminated':False
#         }},
#         Action.EXPLAIN.value:{
#             True:{
#                 'answer':"the explain content",
#                 'reward':0.5,
#                 'terminated':False
#             },
#             False:{
#                 'answer':"it is obviously, but I will try explain it too",
#                 'reward': -0.5,
#                 'terminated':False
#         }}
#         }

# terminated_table = [[True, True],
#                     [True, False],
#                     [True, False],
#                     [False, False]]


# answer = [[''," you misunderstdood my words, I mean..."],
#           ['Yes, it is', 'No, it is not'],
#           [None,"none of these"],
#           ["the explain content","it is obviously, but I will try explain it too",]]