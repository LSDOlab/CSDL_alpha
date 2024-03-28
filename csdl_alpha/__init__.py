__version__ = '0.0.0'


# from .manager import get_current_recorder, print_all_recorders, build_new_recorder

from csdl_alpha.src.operations.add import add
from csdl_alpha.src.operations.mult import mult
from csdl_alpha.src.operations.square import square

from csdl_alpha.src.operations.implicit_operations.solvers import *

from .api import *