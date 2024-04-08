from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.node import Node
from csdl_alpha.utils.inputs import variablize

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

    # Properties for the operation
    properties = {
        'linear': False,
        'elementwise': False,
        'diagonal_jacobian': False,
        'convex': False,
        'elementary': True,
        'supports_sparse': False,
        'contains_subgraph':True
    }

    def __init__(self, *args, metadata = None, **kwargs) -> None:
        super().__init__()

        inputs = []
        for arg in args:
            inputs.append(variablize(arg))

        # ordered CSDL input variables
        self.inputs:list = inputs
        self.num_inputs = len(self.inputs)

        # ordered CSDL input variables
        self.outputs:list = None
        self.num_outputs = None

        # metadata
        if metadata is None:
            metadata = {}
        self.metadata = metadata

    def set_outputs(self, outputs:list[Variable]):
        """
        outputs of operation can only be specified once
        """
        if self.outputs is not None:
            raise ValueError("Outputs already set")
        for output in outputs:
            if not isinstance(output, Variable):
                raise ValueError("Outputs must be a list of csdl variables!")
        self.outputs = outputs
        self.num_outputs = len(self.outputs)

    def set_dense_outputs(self, shapes:list[tuple]):
        """
        if variables are dense, automatically create variables given shapes
        """
        for shape in shapes:
            if not isinstance(shape, tuple):
                raise ValueError(f"Output shapes must be tuples. {shape} given.")

        self.set_outputs([Variable(shape = shape) for shape in shapes])


    def set_sparse_outputs(self, shapes:list[list, list, tuple]):
        """
        if output variables are sparse, automatically create sparse variables given rows/vals/shapes
        """
        raise NotImplementedError("Sparse outputs not yet implemented")
    
    def _add_to_graph(self):
        self.recorder._add_node(self)
        for input_variable in self.inputs:
            self.recorder._add_edge(input_variable, self)

        for output_var in self.outputs:
            self.recorder._add_edge(self, output_var)

    def set_inline_values(self):
        # DOESNT WORK
        # output_values = self.compute_inline(x.value for x in self.inputs)

        # just in case:
        if self.num_inputs == 1:
            output_values = self.compute_inline(self.inputs[0].value)
        elif self.num_inputs == 2:
            output_values = self.compute_inline(self.inputs[0].value, self.inputs[1].value)
        else:
            output_values = self.compute_inline(*[x.value for x in self.inputs])

        if self.num_outputs == 1:
            self.outputs[0].value = output_values
        else:
            for output, value in zip(self.outputs, output_values):
                output.value = value

    def finalize_and_return_outputs(self, skip_inline = False):
        """
        Three things:
        - builds edges between inputs to op and outputs to op
        - computes values inline if necessary
        - returns output variables
        """
        self._add_to_graph()

        # if we're computing inline:
        if self.recorder.inline:
            if not skip_inline:
                self.set_inline_values()

        if len(self.outputs) == 1:
            return self.outputs[0]
        else:
            return tuple(self.outputs)

    def compute_inline(self, *args):
        raise NotImplementedError('not implemented') 

    def compute_jax(self, *args):
        raise NotImplementedError('not implemented') 

    def evaluate_jacobian(self, *args):
        raise NotImplementedError('not implemented') 

    def evaluate_sparse_jacobian(self, *args):
        raise NotImplementedError('not implemented') 

    def evaluate_jvp(self, *args):
        raise NotImplementedError('not implemented')

    def evaluate_vjp(self, *args):
        raise NotImplementedError('not implemented')
    

def set_properties(**kwargs):
    """set properties for an operation class"""
    for property, value in kwargs.items():
        if not isinstance(property, str):
            raise ValueError("Property names must be strings")
        if property not in Operation.properties:
            raise ValueError(f"Property {property} not recognized. Must be one of {Operation.properties.keys()}")
        if not isinstance(value, bool):
            raise ValueError("Property values must be boolean")
        
    def decorator(cls):
        properties = cls.properties.copy()
        for property, value in kwargs.items():
            properties[property] = value
        cls.properties = properties
        return cls
    return decorator
