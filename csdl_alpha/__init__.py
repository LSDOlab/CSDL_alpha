__version__ = '0.0.0'


# from .manager import get_current_recorder, print_all_recorders, build_new_recorder

from csdl_alpha.src.operations import *

from csdl_alpha.src.operations.implicit_operations.solvers import *
from csdl_alpha.src.operations.linalg import linear_solvers as linear_solvers

from .api import *