
from .sc import StateConditioned_Model
# from .gc import GoalConditioned_Model
from .sc_div import StateConditioned_Diversity_Model
# from .gc_div import GoalConditioned_Diversity_Model
from .gc_div_joint import GoalConditioned_Diversity_Joint_Model
from .gc_div_joint_gp import GoalConditioned_GoalPrompt_Model
from .skimo import Skimo_Model
from .simpl import SiMPL_Model
from .base import BaseModel
from .flat_gcsl import Flat_GCSL
from .spirl import SPiRL_Model


MODELS = {
    "sc" : StateConditioned_Model,
    "sc_dreamer" : StateConditioned_Model,
    "simpl" : SiMPL_Model,
    # "gc" : GoalConditioned_Model,
    "sc_div" : StateConditioned_Diversity_Model,
    # "gc_div" : GoalConditioned_Diversity_Model,
    "gc_div_joint" : GoalConditioned_Diversity_Joint_Model,
    "gc_div_joint_gp" : GoalConditioned_GoalPrompt_Model,
    "spirl" : SPiRL_Model,
    "skimo" : Skimo_Model,
    "gc_skimo" : Skimo_Model,
    'flat_gcsl' : Flat_GCSL
}


__all__ = [
    'MODELS',
    'StateConditioned_Model',
    'StateConditioned_Diversity_Model',
    'GoalConditioned_Diversity_Joint_Model',
    'GoalConditioned_GoalPrompt_Model',
    'Skimo_Model',
    'SiMPL_Model',
    'BaseModel',
    ]
