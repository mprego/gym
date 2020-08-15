import pandas as pd
from rl_methods import *

def test_mc_update_values():
    obs = (10, 10, False)
    action = 1
    reward = 2
    q = pd.DataFrame({'me':[10], 'dealer':[10], 'ace':[0], 'action':[1], 'reward':[1], 'num_visits':[3]})
    
    expected_n = 4
    expected_reward = 1.25
    
    new_q = mc_update_values(obs, action, reward, q)
    
    assert expected_n == new_q['num_visits'].values[0], 'Wrong n value'
    assert expected_reward == new_q['reward'].values[0], 'Wrong reward value'
    
    
def test_mc_update_policy():
    obs = (10, 10, False)
    action = 1
    reward = 2
    q = pd.DataFrame({'me':[10, 10], 'dealer':[10, 10], 'ace':[0, 0], 'action':[0, 1], 'reward':[-1, 1], 'num_visits':[3, 3]})
    p = pd.DataFrame({'me':[10], 'dealer':[10], 'ace':[0], 'action':[0]})
    
    expected_action = 1
    
    new_p = mc_update_policy(obs, q, p)
    
    assert expected_action == new_p['action'].values[0], 'Wrong action'  
    
    
def test_q_update_values_hit():
    obs = (10, 10, False)
    action = 1
    reward = 2
    q = pd.DataFrame({'me':[10, 10, 14, 14], 'dealer':[10, 10, 10, 10], 'ace':[0, 0, 0, 0], 'action':[0, 1, 0, 1], 'reward':[-1, 1, 0, 3]})
    p = pd.DataFrame({'me':[10], 'dealer':[10], 'ace':[0], 'action':[0]})
    alpha = .25
    
    expected_reward = 2
    
    new_q = q_update_values(obs, action, reward, q, p, alpha)
    q_reward = new_q.loc[(new_q['me']==obs[0]) & (new_q['dealer']==obs[1]) & (new_q['ace']==obs[2]) & (new_q['action']==action)]
      
    assert len(q_reward) == 1, 'Not one and only one row for q df'
    assert expected_reward == q_reward['reward'].values[0], ('Wrong reward value', expected_reward, 'vs', q_reward['reward'].values[0])
    
    
   
def test_q_update_values_stay():
    obs = (10, 10, False)
    action = 0
    reward = 2
    q = pd.DataFrame({'me':[10, 10, 14, 14], 'dealer':[10, 10, 10, 10], 'ace':[0, 0, 0, 0], 'action':[0, 1, 0, 1], 'reward':[-1, 1, 0, 3]})
    p = pd.DataFrame({'me':[10], 'dealer':[10], 'ace':[0], 'action':[0]})
    alpha = .25
    
    expected_reward = -.25
    
    new_q = q_update_values(obs, action, reward, q, p, alpha)
    q_reward = new_q.loc[(new_q['me']==obs[0]) & (new_q['dealer']==obs[1]) & (new_q['ace']==obs[2]) & (new_q['action']==action)]
      
    assert len(q_reward) == 1, 'Not one and only one row for q df'
    assert expected_reward == q_reward['reward'].values[0], ('Wrong reward value', expected_reward, 'vs', q_reward['reward'].values[0])    
    
    
test_mc_update_values()
test_mc_update_policy()
test_q_update_values_hit()
test_q_update_values_stay()