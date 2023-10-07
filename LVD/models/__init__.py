
from .sc import StateConditioned_Model
# from .gc import GoalConditioned_Model
from .sc_div import StateConditioned_Diversity_Model
# from .gc_div import GoalConditioned_Diversity_Model
from .ours_sep_gp import GoalConditioned_GoalPrompt_Model
from .skimo import Skimo_Model
from .simpl import SiMPL_Model
from .base import BaseModel
from .flat_gcsl import Flat_GCSL
from .spirl import SPiRL_Model
from .ours_sep import GoalConditioned_Diversity_Sep_Model
from .ours import GoalConditioned_Diversity_Model
from .ours_sep_short import Ours_Shortskill
from .ours_sep_long import Ours_LongSkill
from .wgcsl import WGCSL
from .ris import RIS

MODELS = {
    "sc" : StateConditioned_Model,
    "sc_dreamer" : StateConditioned_Model,
    "simpl" : SiMPL_Model,
    # "gc" : GoalConditioned_Model,
    "sc_div" : StateConditioned_Diversity_Model,
    # "gc_div" : GoalConditioned_Diversity_Model,
    "ours_gp" : GoalConditioned_GoalPrompt_Model,
    "spirl" : SPiRL_Model,
    "skimo" : Skimo_Model,
    "gc_skimo" : Skimo_Model,
    'flat_gcsl' : Flat_GCSL,
    "ours_sep" : GoalConditioned_Diversity_Sep_Model,
    'ours' : GoalConditioned_Diversity_Model,
    "ours_sep_short" : Ours_Shortskill,
    "ours_sep_long" : Ours_LongSkill,
    
    'flat_wgcsl' : WGCSL,
    'flat_ris' : RIS
}


__all__ = [
    'MODELS',
    'StateConditioned_Model',
    'StateConditioned_Diversity_Model',
    'GoalConditioned_GoalPrompt_Model',
    "GoalConditioned_Diversity_Sep_Model",
    'GoalConditioned_Diversity_Model',
    'Skimo_Model',
    'SiMPL_Model',
    'BaseModel',
    'WGCSL',
    'RIS'
    ]
