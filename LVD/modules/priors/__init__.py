from .sc import StateConditioned_Prior
from .gc import GoalConditioned_Prior
from .sc_div import StateConditioned_Diversity_Prior
from .ours_sep_gp import GoalConditioned_GoalPrompt_Prior
from .skimo import Skimo_Prior
from .simpl import SiMPL_Prior
from .flat_gcsl import Flat_GCSL
from .spirl import SPiRL_Prior
from .ours_sep import GoalConditioned_Diversity_Sep_Prior
from .ours import GoalConditioned_Diversity_Prior
from .ours_short import Ours_Short_Prior
from .ris import RIS_Prior
from .ours_long import Ours_LongSkill_Prior


PRIOR_WRAPPERS = {
    "sc" : StateConditioned_Prior,
    "gc" : GoalConditioned_Prior,
    "sc_div" : StateConditioned_Diversity_Prior,
    "gc_div_joint_gp" : GoalConditioned_GoalPrompt_Prior,
    "spirl" : SPiRL_Prior,
    "skimo" : Skimo_Prior,
    "simpl" : SiMPL_Prior,
    "flat_gcsl" : Flat_GCSL,
    "ours_sep" : GoalConditioned_Diversity_Sep_Prior,
    "ours" : GoalConditioned_Diversity_Prior,
    "ours_short" : Ours_Short_Prior,
    "ours_long" : Ours_LongSkill_Prior,    
    "flat_ris" : RIS_Prior

}

__all__ = ["PRIOR_WRAPPERS"]