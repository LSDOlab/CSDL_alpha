from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.graph import Graph, is_variable
from csdl_alpha.utils.printing import print_tabularized

import numpy as np

def verify_derivatives_inline(
        ofs:list[Variable],
        wrts:list[Variable],
        step_size:float,
        of_wrt_meta_data:dict = None,
        print_results:bool = True,
        raise_on_error:bool = True,
    )->None:
    raise NotImplementedError("use `csdl.derivative_utils.verify_derivatives` instead")

def get_uncontract_action(
        expand_to_shape:tuple[int],
        contraction_axes:tuple[int],
        )->str:
    """
    For vector-jacobianing, expand the given vjp back to an input shape.
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    action = ''
    for i in range(len(expand_to_shape)):
        if i in contraction_axes:
            continue
        action += alphabet[i]
    action += '->'
    for i in range(len(expand_to_shape)):
        action += alphabet[i]
    return action