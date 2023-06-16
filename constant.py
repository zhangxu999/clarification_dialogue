reward_table = {
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
                'terminated':False
        }}
        }
