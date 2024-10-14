# bin/bash
python -u -m habitat_baselines.run \
    --config-name=social_rearrange/llm_fetch_stretch_manipulation.yaml \
    habitat.dataset.data_path=data/datasets/hssd_scene/108736689_177263340/data_123.json.gz \
    habitat.seed=304 \
    habitat_baselines.image_dir="video_dir/108736689_177263340" \
    habitat_baselines.video_dir="video_dir/108736689_177263340" \

#运行机制：habitat_mas_evaluator中调用agent.actor_critic.act执行，此处方法是HierarchicalPolicy的方法
#HierarchicalPolicy的方法act中_update_skills()，分别对上下层的skill都进行更新
#_update_skills()方法中self._high_level_policy.get_next_skill()调用着hl_xxx
#方法（对应xxx_policy.py）中
#现在想的是用llm_policy中的方法，然后在get_next_skill中判断是agent_0或者agent_1，
#分配对应的"target_obj"，在OracleNavPolicy作符号判断，就可以传标志位给OracleNavDiffAction

