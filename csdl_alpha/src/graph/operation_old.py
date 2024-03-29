from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.node import Node
import numpy as np

class Operation(Node):
    """
    Base Operation class.

    Attributes
    ----------
    name : string
        Name of the operation.
    inputs : list
        List of csdl variables.
    outputs : list
        List of csdl variables.
    """


    def __init__(self, *args, metadata = None, **kwargs) -> None:
        super().__init__()
        # check all args are CSDL variables
        if not all(isinstance(arg, (Variable, int, float, np.ndarray)) for arg in args):
            raise ValueError("All args must be either Variable instances or convertible to Variable instances")

        self.name = 'op'

        # Properties for the operation
        self.properties = {
            'linear': False,
            'elementwise': False,
            'diagonal_jacobian': False,
            'convex': False,
            'elementary': True,
            'supports_sparse': False,
        } 

        # ordered CSDL input variables
        self.inputs:list = args

        # ordered CSDL input variables, convert int/float/ndarray to Variable
        # self.inputs:list = [Variable(value=arg) if isinstance(arg, (int, float, np.ndarray)) else arg for arg in args]

        # ordered CSDL output variables (filled later by add_outputs/add_outputs_shapes)
        self.outputs:list = None

        # metadata
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    def set_output_shapes(self, *shapes:tuple[int]):
        if self.outputs is not None:
            raise ValueError("Outputs already been assigned")

        for shape in shapes:
            if not isinstance(shape, tuple):
                raise ValueError("Output shapes must be tuples")
        self.outputs = [Variable(shape = shape) for shape in shapes]

    def set_outputs_shape_type(self, *vars: Variable):
        if self.outputs is not None:
            raise ValueError("Outputs already been assigned")
        for var in vars:
            if not isinstance(var, Variable):
                raise ValueError(f"var must be a Variable. {var} given")
        self.outputs = vars

    def get_outputs(self):

        self.recorder._add_node(self)
        for input_variable in self.inputs:
            self.recorder._add_edge(input_variable, self)

        for output_var in self.outputs:
            self.recorder._add_edge(self, output_var)

        # if we're computing inline:
        if self.recorder.inline:
            self.set_inline_values()

        if len(self.outputs) == 1:
            return self.outputs[0]
        else:
            return tuple(self.outputs)
        

    def set_inline_values(self):
        output_values = self.compute_inline(*self.inputs)

        if len(self.outputs) == 1:
            self.outputs[0].value = output_values
        else:
            for output, value in zip(self.outputs, output_values):
                output.value = output_values

    def compute_inline(self, *args):
        raise NotImplementedError('not implemented') 
    
    def evaluate_jacobian(self, *args):
        raise NotImplementedError('not implemented') 

    def evaluate_sparse_jacobian(self, *args):
        raise NotImplementedError('not implemented') 

    def evaluate_jvp(self, *args):
        raise NotImplementedError('not implemented')

    def evaluate_vjp(self, *args):
        raise NotImplementedError('not implemented')

class ElementwiseOperation(Operation):

    def __init__(self,*args, **kwargs):
        
        super().__init__(*args, **kwargs)

        self.properties['elementwise'] = True
        self.properties['diagonal_jacobian'] = True

        out_shape = (args[0].shape,)
        self.set_output_shapes(out_shape)


class ComposedOperation(Operation):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.properties['elementary'] = False

        self.recorder._enter_subgraph()
        
        for input in self.inputs:
            self.recorder._add_node(input)
        outputs = self.evaluate_composed(*args)

        if isinstance(outputs, tuple):
            self.outputs = outputs
        else:
            self.outputs = [outputs]
        self.graph = self.recorder.active_graph

        self.recorder._exit_subgraph()

        for output in self.outputs:
            self.recorder._add_node(output)

    def get_outputs(self):
        outputs = super().get_outputs()
        return outputs

    
    def set_inline_values(self):
        pass

class SparseOperation(ComposedOperation):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.properties['supports_sparse'] = True