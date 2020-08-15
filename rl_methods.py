import numpy as np
import pandas as pd

def find_possible_next_states(obs, action, df):
    if action == 0:
        return df[0:0]
    
    else:
        # dealer card won't change
        df2 = df.loc[df['dealer']==obs[1]]

        if obs[2] == 0:
            # if no usable ace and draw normal card or non usable ace
            ns_1 = df2.loc[(df2['me']>=obs[0]+1) & (df2['me']<=obs[0]+10) & (df2['ace']==0)]
            # if no usable ace and draw usable ace
            ns_2 = df2.loc[(df2['me']==obs[0]+11) & (df2['ace']==1)]
        else:
            # if usable ace and keep usable ace status
            ns_1 = df2.loc[(df2['me']>=obs[0]+1) & (df2['me']<=obs[0]+10) & (df2['ace']==1)]
            # if usable ace and lose usable ace status
            ns_2 = df2.loc[(df2['me']>=obs[0]+1-10) & (df2['me']<=obs[0]+10-10) & (df2['ace']==0)] 
        return ns_1.append(ns_2)

def mc_update_values(obs, action, reward, q):
    q_row = q.loc[(q['me']==obs[0]) & (q['dealer']==obs[1]) & (q['ace']==obs[2]) & (q['action']==action)]
    n = q_row['num_visits'].values[0]
    old_reward = q_row['reward'].values[0]
    new_reward = (old_reward*n + reward)/(n+1)
    new_q = q#.copy()
    new_q.loc[(new_q['me']==obs[0]) & (new_q['dealer']==obs[1]) & (new_q['ace']==obs[2]) & (new_q['action']==action),'reward'] = new_reward
    new_q.loc[(new_q['me']==obs[0]) & (new_q['dealer']==obs[1]) & (new_q['ace']==obs[2]) & (new_q['action']==action), 'num_visits'] = n + 1
    return new_q

def mc_update_policy(obs, q, p):
    options = q.loc[(q['me']==obs[0]) & (q['dealer']==obs[1]) & (q['ace']==obs[2])].copy()
    if len(options) > 0:
        options = options.sort_values('reward', ascending=False).reset_index()
        action = options.loc[0, 'action']
        new_p = p#.copy()
        new_p.loc[(p['me']==obs[0]) & (p['dealer']==obs[1]) & (p['ace']==obs[2]), 'action'] = action
    return new_p


def q_update_values(obs, action, reward, q, p, alpha):
    q_row = q.loc[(q['me']==obs[0]) & (q['dealer']==obs[1]) & (q['ace']==obs[2]) & (q['action']==action)]
    old_reward = q_row['reward'].values[0]
    # determine max value of next states
    possible_next_states = find_possible_next_states(obs, action, q)
    if len(possible_next_states) > 0:
        max_action_reward = np.max(possible_next_states['reward'])
    else:
        max_action_reward = 0

    update_reward = reward + max_action_reward
    new_reward = old_reward + alpha * (update_reward - old_reward)
    new_q = q#.copy()
    new_q.loc[(new_q['me']==obs[0]) & (new_q['dealer']==obs[1]) & (new_q['ace']==obs[2]) & (new_q['action']==action),'reward'] = new_reward
    return new_q

def q_update_policy(obs, q, p):
    options = q.loc[(q['me']==obs[0]) & (q['dealer']==obs[1]) & (q['ace']==obs[2])].copy()
    if len(options) > 0:
        options = options.sort_values('reward', ascending=False).reset_index()
        action = options.loc[0, 'action']
        new_p = p#.copy()
        new_p.loc[(p['me']==obs[0]) & (p['dealer']==obs[1]) & (p['ace']==obs[2]), 'action'] = action
    return new_p    
    
def choose_action(obs, p, eps=0):
    p_row = p.loc[(p['me']==obs[0]) & (p['dealer']==obs[1]) & (p['ace']==obs[2])]    
    action = int(p_row['action'].values[0])
    
    if np.random.randint(0, 1) < eps:
        return np.random.randint(0,2)
    else:
        return action  
    