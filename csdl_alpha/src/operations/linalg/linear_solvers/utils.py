from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, validate_and_variablize, get_type_string
from csdl_alpha.src.operations.linalg.utils import process_matA_vecb, check_vecB

import numpy as np
from typing import Callable, Union


def build_matvec_subgraph(A:Callable, b:Variable, recorder, name):
    assert callable(A), f"A must be a callable, but got {type(A)}"

    # Enter subgraph of matvec loop
    recorder._enter_subgraph(name=name, add_missing_variables=True)
    v = Variable(name = 'v', shape = (b.size,), value = np.zeros((b.size,)))
    A_v = A(v).reshape((b.size,))
    matvec_graph = recorder.active_graph
    recorder._exit_subgraph()

    inputs = []
    for input_var in matvec_graph.inputs:
        inputs.append(input_var)

    return matvec_graph, inputs, A_v, v

def process_linsolve_A_b(
        A:Union[Variable,Callable],
        b:Variable,
    ):
    b = validate_and_variablize(b)
    b_orig = b
    if isinstance(A, Variable):
        A = validate_and_variablize(A, raise_on_sparse=False)
        A,b = process_matA_vecb(A,b)
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matrix A must be square, but has shape {A.shape}")
        return A, b, False, b_orig.shape
    elif callable(A):
        b = check_vecB(b)
        return A, b, True, b_orig.shape
    else:
        raise TypeError(f'A must be either a Variable or a function f such that A@v == f(v). Type {get_type_string(A)} given.')

def return_b(output_b:Variable, orig_b_shape)->Variable:
    if len(orig_b_shape) == 2:
        return output_b
    if len(orig_b_shape) == 1:
        return output_b.reshape((output_b.size,))