from .base import BaseModel
from .flat_gcsl import Flat_GCSL
from .wgcsl import WGCSL

from .spirl import SPiRL_Model
from .skimo import SkiMo_Model
from .ours import Ours_Model
from .ours_short import Ours_Shortskill
from .ours_long import Ours_LongSkill

MODELS = {
    'flat_gcsl' : Flat_GCSL,
    'flat_wgcsl' : WGCSL,
    "gc_spirl" : SPiRL_Model,
    "gc_skimo" : SkiMo_Model,
    "ours_sep" : Ours_Model,
    "ours_sep_short" : Ours_Shortskill,
    "ours_sep_long" : Ours_LongSkill,
}

__all__ = [
    'MODELS',
    'BaseModel',
    'Flat_GCSL',
    'WGCSL',
    'SPiRL_Model',
    'SkiMo_Model',
    'Ours_Model',
    "Ours_Shortskill",
    "Ours_LongSkill",
    ]
