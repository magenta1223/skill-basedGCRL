from .flat_gcsl import Flat_GCSL
from .spirl import SPiRL_Prior
from .skimo import SkiMo_Prior
from .ours import Ours_Prior
from .ours_short import Ours_Short_Prior
from .ours_long import Ours_LongSkill_Prior


PRIOR_WRAPPERS = {
    "flat_gcsl" : Flat_GCSL, # for gcsl and wgcsl
    "spirl" : SPiRL_Prior,
    "skimo" : SkiMo_Prior,
    "ours" : Ours_Prior,
    "ours_short" : Ours_Short_Prior,
    "ours_long" : Ours_LongSkill_Prior,    
}

__all__ = ["PRIOR_WRAPPERS"]