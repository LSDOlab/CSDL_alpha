from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
from csdl_alpha.src.operations.operation_subclasses import ComposedOperation
import csdl_alpha.utils.test_utils as csdl_tests
from csdl_alpha.utils.typing import VariableLike

import numpy as np
from csdl_alpha.src.operations.tensor.expand import expand as csdl_expand
from csdl_alpha.src.operations.sum import sum as csdl_sum


class Einsum(ComposedOperation):
    '''
    Einstein summation of a list of Variables according to the specified subscripts.
    '''
    def __init__(self, *args, expansion_actions, expanded_shape, summation_axes):
        super().__init__(*args)
        self.name = 'einsum'
        self.expansion_actions = expansion_actions
        self.expanded_shape    = expanded_shape
        self.summation_axes    = summation_axes
        self.ord   = ord

    def evaluate_composed(self, *args):
        actions    = self.expansion_actions
        axes       = self.summation_axes
        exp_shape  = self.expanded_shape
        
        return evaluate_einsum(args, actions, exp_shape, axes)
    
def evaluate_einsum(args, actions, exp_shape, axes):
    # TODO: Replace this with 
    #       csdl.prod(*[csdl_expand(arg, exp_shape, action) for arg, action in zip(args, actions)]) 
    #       later
    product    = csdl_expand(args[0], exp_shape, actions[0])
    for arg, action in zip(args[1:], actions[1:]):
        product = product * csdl_expand(arg, exp_shape, action)
    out = csdl_sum(product, axes=axes)

    return out
    

def einsum(*args, action=None)->Variable:
    """
    Einstein summation of a list of Variables according to the specified subscripts.

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

    if len(args) == 1:
        raise ValueError('At least two variables must be provided as input for einsum. \
                         For a single input variable A, use more efficient csdl.sum(A) \
                         for summation along specified axes.')

    if action is None:
        raise ValueError('Cannot perform einsum without "action" specified.')
    
    if not isinstance(action, str):
        raise ValueError('"action" must be a string.')
    
    args = [variablize(x) for x in args]
    arg_strings, out_string = action.split('->')
    arg_strings = arg_strings.split(',')

    if len(arg_strings) != len(args):
        raise ValueError('Number of input Variables does not match the number of input strings in the action.')
    
    for i, arg, arg_str in zip(len(args), args, arg_strings):
        if len(arg.shape) != len(arg_str):
            raise ValueError("{i}th input tensor's shape {arg.shape} does not match the number of dimensions in the \
                             {i}th input {arg_str} in the specified `action` string.")
        
        if not all(arg_str.count(char) == 1 for char in arg_str):
            raise ValueError('Each character in the input string must appear exactly once.')
        
    concatenated_arg_strings = ''.join(arg_strings)
    unique_axes_str = ''.join(set('concatenated_arg_strings'))
    if not all(out_string.count(char) == 1 for char in out_string):
        raise ValueError('Each character in the output string must appear exactly once.')
    if not all(char in unique_axes_str for char in out_string):
        raise ValueError('Each character in the output string must appear in the input strings.')
    
    

    expansion_actions = [arg_str + '->' + unique_axes_str for arg_str in arg_strings]
    out_shape         = tuple([arg.shape[arg_str.index(char)] for char in out_string])
    summation_axes    = tuple([unique_axes_str.index(char) for char in out_string])
    reorder_axes_str  = '->'.join([summed_str, out_string])

    # TODO: Compute expanded_shape
    expanded_shape = tuple([arg.shape[arg_str.index(char)] for char in unique_axes_str])

    if not all(out_str.count(char) == 1 for char in in_str):
        raise ValueError('Each character in the input string must appear exactly once in the output string.')
    
    if in_shape != tuple([out_shape[out_str.index(char)] for char in in_str]):
        raise ValueError('Input tensor shape is not compatible with the output shape specified in the action.')
    
    ones_str   = ''.join([char for char in out_str if char not in in_str])
    ones_shape = tuple([out_shape[out_str.index(char)] for char in ones_str])
    einsum_str = in_str + ',' + ones_str + '->' + out_str
            
    op = Einsum(*args, expansion_actions, expanded_shape, summation_axes)

    return op.finalize_and_return_outputs()