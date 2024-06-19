from csdl_alpha.src.graph.operation import Operation, set_properties
# from csdl_alpha.src.graph.graph import Graph
import numpy as np
from csdl_alpha.utils.inputs import variablize
from csdl_alpha.src.graph.variable import Variable, Constant

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
        from csdl_alpha.src.graph.graph import Graph
        if self._subgraph is not None:
            raise ValueError("Subgraph already set")
        self._subgraph:Graph = graph
    
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
        
        unique_args = set()
        new_args = []
        import csdl_alpha as csdl
        for arg in args:
            arg = variablize(arg)
            if arg in unique_args:
                new_arg = csdl.copyvar(arg)
                new_args.append(new_arg)
                unique_args.add(new_arg)
            else:
                unique_args.add(arg)
                new_args.append(arg)
        
        super().__init__(*new_args, **kwargs)

    def evaluate_composed(self, *args):
        raise NotImplementedError("Composed operations must implement the evaluate_composed method")
    
    def compute_jax(self, *args):
        from csdl_alpha.backends.jax.graph_to_jax import create_jax_function
        jax_fn = create_jax_function(self.get_subgraph(), self.outputs, self.inputs)
        return tuple(jax_fn(*args))

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
    
    def set_inline_values(self, debug = False):
        self.get_subgraph().execute_inline(debug=debug)

        keep_set = set(self.outputs).union(set(self.inputs))
        subgraph = self.get_subgraph()
        for node in subgraph.node_table:
            if isinstance(node, Variable):

                do_not_delete = False
                if subgraph.in_degree(node) == 0:
                    do_not_delete = True
                if isinstance(node, Constant):
                    do_not_delete = True
                
                for pred in subgraph.predecessors(node):
                    if isinstance(pred, SubgraphOperation):
                        do_not_delete = True

                if not do_not_delete:
                   if node not in keep_set:
                        # print([node2 for node2 in keep_set])
                        node.value = None

    def evaluate_vjp(self, cotangents, *inputs_outputs):
        # TODO: extremely messy and crappy. FIX!
        # TODO: extremely messy and crappy. FIX!
        # TODO: extremely messy and crappy. FIX!
        # TODO: extremely messy and crappy. FIX!
        # TODO: extremely messy and crappy. FIX!
        # TODO: extremely messy and crappy. FIX!
        # TODO: extremely messy and crappy. FIX!
        # TODO: extremely messy and crappy. FIX!!!!
        # TODO: extremely messy and crappy. FIX
        # TODO: extremely messy and crappy. FIX!
        # TODO: extremely messy and crappy. FIX!
        # TODO: extremely messy and crappy. FIX!

        # Created a new composed operation which inputs the same inputs as the original operation
        # plus the cotangents. This composed operation just computes the single VJP operation.
        inputs = inputs_outputs[:self.num_inputs]
        outputs = inputs_outputs[self.num_inputs:]
        import csdl_alpha as csdl

        # Find all wrts.
        wrts = []
        for input_var in inputs:
            if cotangents.check(input_var):
                wrts.append(input_var)

        # All the inputs we pass into the composed operation
        composed_inputs = []
        for of in outputs:
            cotangents.initialize(of)
            if cotangents[of] is None:
                cotangents.accumulate(of, csdl.Variable(value = np.zeros(of.shape)))
            composed_inputs.append(cotangents[of]) # Cotangents we need to propagate by
        for orig_input in inputs:
            composed_inputs.append(orig_input) # All the original inputs
        num_cots = len(outputs)

        rec = csdl.get_current_recorder()
        # rec.visualize_graph()

        from csdl_alpha.src.operations.derivatives.reverse import vjp
        # This is the function that gets executed within the composed operation
        # It takes the cotangents and the original inputs
        # We first re-compute the composed operation and then compute the VJP again
        def composed_vjp(*cotangets_and_inputs):
            output_cotangents = cotangets_and_inputs[:num_cots]
            original_inputs = cotangets_and_inputs[num_cots:]

            outputs_again = self.evaluate_composed(*original_inputs)
            
            if not isinstance(outputs_again, tuple):
                outputs_again = (outputs_again,)

            seeds = []
            for i, output_var in enumerate(outputs_again):
                seeds.append((output_var, output_cotangents[i]))
            
            rec = csdl.get_current_recorder()

            wrts_composed = vjp(seeds, wrts, rec.active_graph)
            
            outputs_composed = []
            for wrt_composed, wrt_cotangent in wrts_composed.items():
                if wrt_cotangent is None:
                    zeros = csdl.Variable(value = np.zeros(wrt_composed.shape))
                    outputs_composed.append(zeros)
                else:
                    outputs_composed.append(csdl.copyvar(wrt_cotangent))
                # print('IBE', wrt_composed.shape, wrt_cotangent.shape)

            outputs_composed = tuple(outputs_composed)
            if len(outputs_composed) == 1:
                return outputs_composed[0]
            else:
                return outputs_composed

        name = self.name

        class VJPInputsOutputs(ComposedOperation):
            def __init__(self, *args):
                super().__init__(*args)
                self.name = f'vjp_{name}'
            def evaluate_composed(self, *args):
                outs = composed_vjp(*args)
                return outs
            
        wrt_derivs = VJPInputsOutputs(*composed_inputs).finalize_and_return_outputs()
        if not isinstance(wrt_derivs, tuple):
            wrt_derivs = (wrt_derivs, )

        #rec.visualize_graph(format = 'png')
        i = 0
        for input_var in inputs:
            if cotangents.check(input_var):
                cotangents.accumulate(input_var, wrt_derivs[i])
                i+=1


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