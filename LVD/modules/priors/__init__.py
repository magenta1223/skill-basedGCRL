from .sc import StateConditioned_Prior
from .gc import GoalConditioned_Prior
from .sc_div import StateConditioned_Diversity_Prior
from .gc_div import GoalConditioned_Diversity_Prior
from .gc_div_joint import GoalConditioned_Diversity_Joint_Prior
from .gc_div_joint_gp import GoalConditioned_GoalPrompt_Prior
from .skimo import Skimo_Prior
from .simpl import SiMPL_Prior
from .flat_gcsl import Flat_GCSL
from .spirl import SPiRL_Prior
from .gc_div_joint_sep import GoalConditioned_Diversity_Joint_Sep_Prior

PRIOR_WRAPPERS = {
    "sc" : StateConditioned_Prior,
    "gc" : GoalConditioned_Prior,
    "sc_div" : StateConditioned_Diversity_Prior,
    "gc_div" : GoalConditioned_Diversity_Prior,
    "gc_div_joint" : GoalConditioned_Diversity_Joint_Prior,
    "gc_div_joint_gp" : GoalConditioned_GoalPrompt_Prior,
    "spirl" : SPiRL_Prior,
    "skimo" : Skimo_Prior,
    "simpl" : SiMPL_Prior,
    "flat_gcsl" : Flat_GCSL,
    "gc_div_joint_sep" : GoalConditioned_Diversity_Joint_Sep_Prior
}

__all__ = ["PRIOR_WRAPPERS"]