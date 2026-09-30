from intellectagent.rl_week7 import OfflineOutcome, OnlineFeedback, offline_reward, online_reward

def test_offline_and_online_rewards_are_separate():
    good=offline_reward(OfflineOutcome(1,False,False,True,100))
    bad=offline_reward(OfflineOutcome(None,False,True,False,100))
    assert good > bad
    assert online_reward(OnlineFeedback(thumb=1,source_opened=True,reasked=False)) > 0
    assert online_reward(OnlineFeedback(thumb=-1,source_opened=False,reasked=True)) < 0
