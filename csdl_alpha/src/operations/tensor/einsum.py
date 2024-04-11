from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
import csdl_alpha.utils.test_utils as csdl_tests
import pytest



def einsum(
        *operands: list[Variable],
        subscripts: str,
        partial_format='dense'
    )->Variable:
    """_summary_

    Parameters
    ----------
    subscripts : str
        _description_
    partial_format : str, optional
        _description_, by default 'dense'

    Returns
    -------
    Variable
        _description_
    """


    operands_list = []
    for operand in operands:
        operands_list.append(variablize(operand))

    

    return
