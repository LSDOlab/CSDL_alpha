# Basic operations
from .add import add
from .mult import mult
from .sub import sub
from .square import square
from .division import div
from .power import power
from .sqrt import sqrt
from .exp import exp
from .log import log
from .reshape import reshape
from .sum import sum
from .average import average
from .maximum import maximum
from .minimum import minimum
from .absolute import absolute

# getting and setting
from .set_get.setindex import set_index
from .set_get.getindex import get_index
from .set_get.loop_slice import _loop_slice as slice


# Trigonometric operations
from .trig import sin, cos, tan

# Linear algebra operations
from .block_matrix import blockmat
from .linalg.matvec import matvec
from .linalg.matmat import matmat

# Tensor operations
from .outer import outer
from .inner import inner