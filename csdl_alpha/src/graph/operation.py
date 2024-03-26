from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.node import Node

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
    output_shapes : list
        List of output variable shapes.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.name = 'op'

        # determined by operation subclass using self.add_output_shape
        self.output_shapes:list = None

        # ordered CSDL input variables
        self.inputs:list = args

        # ordered CSDL output variables (filled later by add_output_shape)
        self.outputs:list = []

        # recorder object
        import csdl_alpha
        self.recorder = csdl_alpha.get_current_recorder()

    def add_output_shapes(self, *shapes):
        if self.output_shapes is not None:
            raise ValueError("Output shapes have already been assigned")

        for shape in shapes:
            if not isinstance(shape, tuple):
                raise ValueError("Output shapes must be tuples")
        self.output_shapes = shapes

    def get_outputs(self):

        self.recorder._add_node(self)
        for input_variable in self.inputs:
            self.recorder._add_edge(input_variable, self)

        for shape in self.output_shapes:
            output_var = Variable(shape)
            
            self.outputs.append(output_var)

            # self.recorder._add_node(output_var)
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

        if not isinstance(output_values, tuple):
            self.outputs[0].value = output_values
        else:
            for output, value in zip(self.outputs, output_values):
                output.value = output_values

    def compute_inline(self, *args):
        raise NotImplementedError('not implemented') 
    
    def evaluate_diagonal_jacobian(self, *args):
        raise NotImplementedError('not implemented') 

    def evaluate_jvp(self, *args):
        raise NotImplementedError('not implemented')

    def evaluate_vjp(self, *args):
        raise NotImplementedError('not implemented')



class ElementwiseOperation(Operation):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        out_shape = (args[0].shape,)
        self.add_output_shapes(out_shape)
