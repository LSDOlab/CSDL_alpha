from csdl_alpha.src.operations.operation_subclasses import ComposedOperation, check_expand_subgraphs
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.typing import VariableLike
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests
import csdl_alpha as csdl

from scipy import sparse as sp
import numpy as np

# TODO: make start_weights, stop_weights csdl variables
class LinearCombination(ComposedOperation):
    def __init__(self, start, stop, start_weights, stop_weights, num_steps):
        super().__init__(start, stop)
        self.name = 'linear_combination'
        self.start_weights = start_weights
        self.stop_weights = stop_weights
        self.num_steps = num_steps

    def evaluate_composed(self, start, stop):
        return evaluate_linear_combination(start, stop, self.start_weights, self.stop_weights, self.num_steps)
    
def evaluate_linear_combination(start, stop, start_weights, stop_weights, num_steps):
    if len(start.shape) == 1:
        num_per_step = start.shape[0]
    else:
        num_per_step = np.prod(start.shape)

    map_num_outputs = num_steps*num_per_step
    map_num_inputs = num_per_step
    map_start = sp.lil_matrix((map_num_outputs, map_num_inputs))
    map_stop = sp.lil_matrix((map_num_outputs, map_num_inputs))
    for i in range(num_steps):
        start_step_map = (sp.eye(num_per_step)) * start_weights[i]
        map_start[i*num_per_step:(i+1)*num_per_step, :] = start_step_map

        stop_step_map = (sp.eye(num_per_step)) * stop_weights[i]
        map_stop[i*num_per_step:(i+1)*num_per_step, :] = stop_step_map

    num_per_step = num_per_step
    map_start = map_start.tocsc()
    map_stop = map_stop.tocsc()

    flattened_start = start.reshape((num_per_step,1))
    flattened_stop = stop.reshape((num_per_step,1))
    mapped_start_array = csdl.sparse.matvec(map_start, flattened_start)
    mapped_stop_array = csdl.sparse.matvec(map_stop, flattened_stop)

    flattened_output = mapped_start_array + mapped_stop_array

    out = flattened_output.reshape((num_steps,) + tuple(start.shape))
    return out

def linear_combination(start:VariableLike, stop:VariableLike, num_steps:int, start_weights:np.ndarray=None, stop_weights:np.ndarray=None)->Variable:
    """Elementwise linear combination of two tensors.

    Parameters
    ----------
    start : VariableLike
        First tensor to be linearly combined.
    stop : VariableLike
        Second tensor to be linearly combined.
    start_weights : np.ndarray
        Weights for the first tensor in the linear combination.
    stop_weights : np.ndarray
        Weights for the second tensor in the linear combination.
    num_steps : int
        Number of steps in the linear combination.

    Returns
    -------
    Variable
        Elementwise linear combination of the two tensors.
    
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> start = csdl.Variable(value = np.array([1.0, 2.0, 3.0]))
    >>> stop = csdl.Variable(value = np.array([4.0, 5.0, 6.0]))
    >>> csdl.linear_combination(start, stop, 3).value
    array([[4. , 4.5, 4. ],
           [2.5, 3.5, 4.5],
           [1. , 2. , 3. ]])
    """
    start = validate_and_variablize(start)
    stop = validate_and_variablize(stop)

    csdl.check_parameter(num_steps, 'num_steps', types=int, lower=1)
    
    
    if start_weights is None:
        if num_steps == 1:
            start_weights = np.array([0.5])
        start_weights = np.linspace(0, 1, num_steps)
    
    if stop_weights is None:
        if num_steps == 1:
            stop_weights = np.array([0.5])
        stop_weights = np.linspace(1, 0, num_steps)
    
    if check_expand_subgraphs():
        return evaluate_linear_combination(start, stop, start_weights, stop_weights, num_steps)
    else:
        op = LinearCombination(start, stop, start_weights, stop_weights, num_steps)
        return op.finalize_and_return_outputs()
    


class TestLinearCombination(csdl_tests.CSDLTest):

    def test_functionality(self):
        self.prep()

        x_val = np.array([1.0, 2.0, 3.0])
        y_val = np.array([1.0, 2.0, 3.0])
        
        x = csdl.Variable(value = x_val)
        y = csdl.Variable(value = y_val)

        compare_values = []

        ls = csdl.linear_combination(x, y, 5)
        ls_val = np.linspace(x_val, y_val, 5)
        compare_values += [csdl_tests.TestingPair(ls, ls_val, tag = 'ls')]

        self.run_tests(compare_values = compare_values)

    # TODO: fix this (ask Mark?)
    # def test_example(self,):
    #     self.docstest(linear_combination)