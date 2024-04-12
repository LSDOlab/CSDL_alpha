from typing import Union
import numpy as np
from csdl_alpha.src.graph.variable import Variable

VariableLike = Union[Variable, np.ndarray, float, int, np.integer]