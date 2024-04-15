from csdl_alpha.src.graph.operation import Operation, set_properties


@set_properties(elementwise = True, diagonal_jacobian = True)
class ElementwiseOperation(Operation):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        out_shapes = (args[0].shape,)
        self.set_dense_outputs(out_shapes)

        for arg in args:
            if arg.shape != args[0].shape:
                raise ValueError("All inputs must have the same shape for elementwise operations")

# Maybe later?
# class BinaryElementwiseBuilder():
#     def __init__(self):
#         self.function_mappings = {}

#     def map(self, sparse_a:bool, sparse_b:bool, scalar_a:bool, scalar_b:bool, operation_class):
#         self.function_mappings[(sparse_a, sparse_b, scalar_a, scalar_b)] = operation_class

#     def build(self, variable_a, variable_b):
#         from csdl_alpha.src.graph.variable import SparseMatrix
#         sparse_a = isinstance(variable_a, SparseMatrix)
#         sparse_b = isinstance(variable_b, SparseMatrix)
#         scalar_a = variable_a.size == 1
#         scalar_b = variable_b.size == 1

#         key = (sparse_a, sparse_b, scalar_a, scalar_b)
#         if key not in self.function_mappings:
#             raise TypeError("No function mapping found for given inputs")
#         else:
#             return self.function_mappings[(sparse_a, sparse_b, scalar_a, scalar_b)]
    
@set_properties(elementary = False, contains_subgraph = True)
class SubgraphOperation(Operation):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self._subgraph = None

    def assign_subgraph(self, graph):
        """
        assigns a subgraph to the operation. Must be called once and can only be called once.
        """
        if self._subgraph is not None:
            raise ValueError("Subgraph already set")
        self._subgraph = graph
    
    def get_subgraph(self):
        """
        get this operation's subgraph
        """
        if self._subgraph is None:
            raise ValueError("Subgraph not set")
        return self._subgraph

@set_properties()
class ComposedOperation(SubgraphOperation):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)

    def finalize_and_return_outputs(self):
        self.recorder._enter_subgraph(name = self.name)
        for input in self.inputs:
            self.recorder._add_node(input)
        outputs = self.evaluate_composed(*self.inputs)

        if isinstance(outputs, tuple):
            self.set_outputs(outputs)
        else:
            self.set_outputs([outputs])

        self.assign_subgraph(self.recorder.active_graph)

        self.recorder._exit_subgraph()

        for output in self.outputs:
            self.recorder._add_node(output)

        outputs = super().finalize_and_return_outputs(skip_inline = True)
        return outputs
    
    def set_inline_values(self):
        self.get_subgraph().execute_inline()

# def expand_subgraph(evaluate_function):
    # """
    # Decorator to expand a composed operation to a flat graph

    # Parameters
    # ----------
    # evaluate_function : function
    #     direct function to evaluate the composed operation
    # """
    # def decorator(func):
    #     from csdl_alpha.api import manager
    #     recorder = manager.active_recorder
    #     if recorder.expand_ops:
    #         return evaluate_function
    #     else:
    #         return func
        
    # return decorator

def check_expand_subgraphs():
    from csdl_alpha.api import manager
    recorder = manager.active_recorder
    return recorder.expand_ops