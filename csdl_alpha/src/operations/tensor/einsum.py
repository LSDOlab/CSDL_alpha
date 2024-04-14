from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
from csdl_alpha.src.operations.operation_subclasses import ComposedOperation
import csdl_alpha.utils.test_utils as csdl_tests
from csdl_alpha.utils.typing import VariableLike

import numpy as np
from csdl_alpha.src.operations.tensor.expand import expand as csdl_expand
from csdl_alpha.src.operations.product import product as csdl_prod
from csdl_alpha.src.operations.sum import sum as csdl_sum


class Einsum(ComposedOperation):
    '''
    Einstein summation of a list of Variables according to the specified subscripts.
    '''
    def __init__(self, *args, exp_actions=None, exp_shape=None, summation_axes=None):
        super().__init__(*args)
        self.name = 'einsum'
        self.expansion_actions = exp_actions
        self.expanded_shape    = exp_shape
        self.summation_axes    = summation_axes

    def evaluate_composed(self, *args):
        actions    = self.expansion_actions
        axes       = self.summation_axes
        exp_shape  = self.expanded_shape
        
        return evaluate_einsum(args, actions, exp_shape, axes)
    
def evaluate_einsum(args, actions, exp_shape, axes):
    # TODO: Either this
    # exp_product    = csdl_expand(args[0], exp_shape, actions[0])
    # for arg, action in zip(args[1:], actions[1:]):
    #     exp_product = exp_product * csdl_expand(arg, exp_shape, action)

    # TODO: Or this [whichever is more efficient]
    expanded_inputs = [csdl_expand(arg, exp_shape, action) for arg, action in zip(args, actions)]
    if len(expanded_inputs) == 1:
        # if there's only one expanded input, taking the product gives the product of all the elements
        # in that input
        exp_product = expanded_inputs[0]
    else:
        exp_product = csdl_prod(*expanded_inputs)
    
    if len(axes) > 0:
        # more efficient summation for inner product
        if len(axes) == len(exp_shape):
            out = csdl_sum(exp_product)
        else:       
            out = csdl_sum(exp_product, axes=axes)
    else:
        # No summation needed if axes=() empty tuple
        out = exp_product

    return out
    

def einsum(*args, action=None)->Variable:
    """
    Einstein summation of a list of Variables according to the specified action.
    The `action` needs to be a string that explicitly specifies the 
    input and output subscripts separated by '->'. 
    The string must contain the explicit indicator '->' to specify the output form.
    For example, if the input Variables are A and B, and the `action` is 'ij,jk->ik',
    the output will be the matrix product of A and B.

    Parameters
    ----------
    args : list of Variables or np.ndarray objects
        Input Variables for Einstein summation.
    action : str
        String specifying the input and output subscripts separated by '->'.
        The input subscripts are separated by commas.
        There must be exactly one output subscript.
        For example, 'ij,jk->ik' specifies the matrix product of two matrices.

    Returns
    -------
    Variable
        Result of Einstein summation of the input Variables according to the specified action.

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = np.array([1, 2, 3]))
    >>> y = csdl.Variable(value = np.array([4, 5]))

    Outer product of x and y:
    
    >>> csdl.einsum(x, y, action='i,j->ij').value
    array([[ 4.,  5.],
           [ 8., 10.],
           [12., 15.]])

    Outer product of x and z:
    
    >>> z = csdl.Variable(value = np.array([[1, 2], [3, 4]]))
    >>> csdl.einsum(x, z, action='i,jk->ijk').value
    array([[[ 1.,  2.],
            [ 3.,  4.]],
    <BLANKLINE>
           [[ 2.,  4.],
            [ 6.,  8.]],
    <BLANKLINE>
           [[ 3.,  6.],
            [ 9., 12.]]])

    Outer product of y and z reordered:

    >>> csdl.einsum(x, z, action='i,jk->kij').value
    array([[[ 1.,  3.],
            [ 2.,  6.],
            [ 3.,  9.]],
    <BLANKLINE>
           [[ 2.,  4.],
            [ 4.,  8.],
            [ 6., 12.]]])

    Dot product of y and z along one axis (same at matrix product z @ y):

    >>> csdl.einsum(y, z, action='j,ij->i').value
    array([14., 32.])

    Inner product of z and t:

    >>> t_np = np.array([[5, 6], [7, 8]])
    >>> csdl.einsum(z, t_np, action='ij,ij->').value
    array([70.])

    Sum of all the elements of z:

    >>> csdl.einsum(z, action='ij->').value
    array([10.])

    Matrix product z @ t:

    >>> csdl.einsum(z, t_np, action='ij,jk->ik').value
    array([[19., 22.],
           [43., 50.]])

    Matrix product z.T @ t:

    >>> csdl.einsum(z, t_np, action='ji,jk->ik').value
    array([[26., 30.],
           [38., 44.]])

    """

    if len(args) == 0:  
        raise ValueError('At least one variable must be provided as input for einsum.')
    
    # TODO: Comment the next 4 lines out later for forcing more efficiency
    # if len(args) == 1:
    #     raise ValueError('At least two variables must be provided as input for einsum. \
    #                      For a single input variable A, use more efficient csdl.sum(A) \
    #                      for summation along specified axes.')

    if action is None:
        raise ValueError('Cannot perform einsum without "action" specified.')
    
    if not isinstance(action, str):
        raise ValueError('"action" must be a string.')
    
    if '->' not in action:
        raise ValueError('Invalid action string. Use "->" to separate the input and output subscripts.')
    
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,->'
    if not all(char in alphabet for char in action):
        raise ValueError(f'The `action` string must contain only valid characters: {alphabet}.')
    
    args = [variablize(x) for x in args]
    arg_strings, out_string = action.split('->')
    arg_strings = arg_strings.split(',')

    if len(arg_strings) != len(args):
        raise ValueError('Number of input Variables does not match the number of input strings in the `action`.')
    
    for i, arg, arg_str in zip(range(len(args)), args, arg_strings):
        if len(arg.shape) != len(arg_str):
            raise ValueError(f"{i}th input tensor's shape {arg.shape} does not match the number of dimensions in the \
                             {i}th input {arg_str} in the specified `action` string.")
        
        if not all(arg_str.count(char) == 1 for char in arg_str):
            raise ValueError('Each character in the input string must appear exactly once.')
        
    concatenated_arg_strings = ''.join(arg_strings)
    unique_axes_str = ''.join(set(concatenated_arg_strings))
    if not all(out_string.count(char) == 1 for char in out_string):
        raise ValueError('Each character in the output string must appear exactly once.')
    if not all(char in unique_axes_str for char in out_string):
        raise ValueError('Each character in the output string must appear in the input strings.')
    
    # exp_str   -> expanded output strings for each input
    # exp_shape -> expanded shape for each input
    # if exp_str = unique_axes_str, then we need to sum along different axes and then reorder
    # that is why the following line is needed to simplify the later operations
    exp_str = out_string + ''.join([char for char in unique_axes_str if char not in out_string])
    exp_shape = [0] * len(exp_str)
    for arg, arg_str in zip(args, arg_strings):
        for j, char in enumerate(arg_str):
            if char in exp_str:
                exp_shape[exp_str.index(char)] = arg.shape[j]

    exp_shape = tuple(exp_shape)

    # exp_actions -> expansion actions for each input
    exp_actions = [arg_str + '->' + exp_str for arg_str in arg_strings]
    out_shape = exp_shape[:len(out_string)]
    summation_axes = tuple(range(len(out_string), len(exp_str)))
    # out_shape         = tuple([arg.shape[arg_str.index(char)] for char in out_string])
    # summation_axes    = tuple([unique_axes_str.index(char) for char in out_string])
    # reorder_axes_str  = '->'.join([summed_str, out_string])

    op = Einsum(*args, exp_actions=exp_actions, exp_shape=exp_shape, summation_axes=summation_axes)

    return op.finalize_and_return_outputs()


class TestEinsum(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = np.array([1, 2, 3])
        y_val = np.array([4, 5])
        z_val = np.array([[1, 2], [3, 4]])
        t_val = np.array([[5, 6], [7, 8]])
        x = csdl.Variable(value = x_val)
        y = csdl.Variable(value = y_val)
        z = csdl.Variable(value = z_val)

        compare_values = []
        # Outer product of x and y:
        s1 = csdl.einsum(x, y, action='i,j->ij')
        t1 = np.einsum('i,j->ij', x_val, y_val)
        compare_values += [csdl_tests.TestingPair(s1, t1, tag='s1')]

        # Outer product of x and z:
        s2 = csdl.einsum(x, z, action='i,jk->ijk')
        t2 = np.einsum('i,jk->ijk', x_val, z_val)
        compare_values += [csdl_tests.TestingPair(s2, t2, tag='s2')]

        # Outer product of y and z reordered:
        s3 = csdl.einsum(x, z, action='i,jk->kij')
        t3 = np.einsum('i,jk->kij', x_val, z_val)
        compare_values += [csdl_tests.TestingPair(s3, t3, tag='s3')]
    
        # Dot product of y and z along one axis (same at matrix product z @ y):
        s4 = csdl.einsum(y, z, action='j,ij->i')
        t4 = np.einsum('j,ij->i', y_val, z_val)
        compare_values += [csdl_tests.TestingPair(s4, t4, tag='s4')]

        # Inner product of z and t:
        s5 = csdl.einsum(z, t_val, action='ij,ij->')
        t5 = np.einsum('ij,ij->', z_val, t_val).flatten()
        compare_values += [csdl_tests.TestingPair(s5, t5, tag='s5')]

        # Sum of all the elements of z:
        s6 =csdl.einsum(z, action='ij->')
        t6 = np.einsum('ij->', z_val).flatten()
        compare_values += [csdl_tests.TestingPair(s6, t6, tag='s5')]
        
        # Matrix product z @ t:
        s7 = csdl.einsum(z, t_val, action='ij,jk->ik')
        t7 = np.einsum('ij,jk->ik', z_val, t_val)
        compare_values += [csdl_tests.TestingPair(s7, t7, tag='s7')]

        # Matrix product z.T @ t:
        s8 = csdl.einsum(z, t_val, action='ji,jk->ik')
        t8 = np.einsum('ji,jk->ik', z_val, t_val)
        compare_values += [csdl_tests.TestingPair(s8, t8, tag='s8')]

        self.run_tests(compare_values = compare_values,)

    def test_docstring(self):
        self.docstest(einsum)

if __name__ == '__main__':
    test = TestEinsum()
    test.test_functionality()
    test.test_docstring()