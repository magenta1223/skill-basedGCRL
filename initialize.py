import json 
from LVD.utils import *
import hydra
from collections import Counter
import pandas as pd


def main():
    # get configs and offline dataset 
    with hydra.initialize("./LVD/configs"):
        cfg = hydra.compose(config_name="ours_kitchen")
    config_parser = ConfigParser()
    cfg = config_parser.parse_cfg(cfg)
    train_dataset = cfg.dataset_cls(cfg, "train")
    sp = StateProcessor("kitchen")


    # training dataset 
    tasks = []
    for seq in train_dataset.seqs:
        subtask_set = set()
        task = ""
        for state in seq.states:
            achieved = sp.state_goal_checker(state)
            if achieved and achieved[-1] not in subtask_set:
                subtask_set.add(achieved[-1])
                task = task + achieved[-1]

        tasks.append(task)

    # initialize transition dictionary 
    transition_dict = {}

    for task in set(tasks):
        task = '<' + task
        for i in range(len(task) - 1):
            transition = task[i:i+2]
            transition_dict[transition] = 0

    # transition count 
    task_counts = Counter(tasks)
    for task, count in task_counts.items():
        task = '<' + task
        for i in range(len(task) - 1):
            transition_dict[task[i:i+2]] += count

    # transition probs
    subtasks = ["<", "M", "K", "B", "T", "L", "S", "H"]
    transition_probs = {}
    for st in subtasks:
        starts_with_st = { transition[1] : count for transition, count in transition_dict.items() if transition[0] == st}
        
        total = sum(starts_with_st.values())
        for k, v in starts_with_st.items():
            starts_with_st[k] = v / total

        transition_probs[st] = starts_with_st

    # save 
    with open("./assets/subtask_transition_prob.json", mode = "w") as f:
        json.dump(transition_probs, f)


    # BFS
    def get_all_tasks(transition_probs, parent_node, task_dict):
        for next_subtask in transition_probs[parent_node[-1]]:
            task = parent_node + next_subtask
            task_dict[task] = task_dict[parent_node] * transition_probs[parent_node[-1]][next_subtask]
            children_tasks = get_all_tasks(transition_probs, task, task_dict)
            for k, v in children_tasks.items():
                task_dict[k] = v
        return task_dict
    kitchen_known_tasks = ['<KBTS','<MKBS','<MKLH','<KTLS',
                '<BTLS','<MTLH','<MBTS','<KBLH',
                '<MKLS','<MBSH','<MKBH','<KBSH',
                '<MBTH','<BTSH','<MBLS','<MLSH',
                '<KLSH','<MBTL','<MKTL','<MKSH',
                '<KBTL','<KBLS','<MKTH','<KBTH']


    tasks = {k : v for k, v in get_all_tasks(transition_probs, "<", {"<" : 1}).items() if len(k) == 5}
    k not in kitchen_known_tasks
    possible_task_4 = pd.DataFrame({"tasks" : tasks.keys(), "probs" : tasks.values()}).sort_values(by = "probs", ascending= False).reset_index(drop = True)
    
    possible_task_4['unseen'] = possible_task_4['tasks'].apply(lambda x : x not in kitchen_known_tasks)

    possible_task_4.to_csv("./assets/possible_tasks.csv", index = False)


if __name__ == "__main__":
    main()