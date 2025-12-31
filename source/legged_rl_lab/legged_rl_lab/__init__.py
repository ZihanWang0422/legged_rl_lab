"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

import os
LEGGED_RL_LAB_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
