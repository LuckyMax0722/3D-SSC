import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
sys.path.append(project_root)
from projects.configs.config import CONF

if CONF.LATENTNET.USE_V3_2:
    from .decoder import Decoder
    from .encoder_v2 import Encoder_x, Encoder_xy
else:
    from .decoder import Decoder
    from .encoder import Encoder_x, Encoder_xy