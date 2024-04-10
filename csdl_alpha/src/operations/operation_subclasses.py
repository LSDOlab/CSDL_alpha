from csdl_alpha.src.graph.operation import Operation, set_properties


@set_properties(elementwise = True, diagonal_jacobian = True)
class ElementwiseOperation(Operation):

    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        out_shapes = (args[0].shape,)
        self.set_dense_outputs(out_shapes)

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
        self.recorder._enter_subgraph()
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