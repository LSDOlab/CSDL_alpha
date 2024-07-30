from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.operations.operation_subclasses import SubgraphOperation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.operations.set_get.loop_slice import _loop_slice as slice
from csdl_alpha.utils.inputs import get_type_string
import numpy as np

def check_batched_variable_pairs(batched_pairs:list[tuple[Variable,Variable]], outer_graph, body_graph):
    batch_size = None

    outer = []
    inner = []
    for pair in batched_pairs:
        outer_batched = pair[0]
        inner_unbatched = pair[1]
        if not isinstance(pair, tuple):
            raise TypeError(f"Expected batched pair to be a tuple, but got {get_type_string(pair)}")
        if len(pair) != 2:
            raise ValueError(f"Expected batched pair to have length 2, but got {len(pair)}")
        if not isinstance(outer_batched, Variable):
            raise TypeError(f"Expected first element of batched pair to be a Variable, but got {get_type_string(outer_batched)}")
        if not isinstance(inner_unbatched, Variable):
            raise TypeError(f"Expected second element of batched pair to be a Variable, but got {get_type_string(inner_unbatched)}")
        
        if len(outer_batched.shape) < 2:
            raise ValueError(f"Expected batched output must be atleast 2D")
        
        if batch_size is None:
            batch_size = outer_batched.shape[0]
        else:
            if outer_batched.shape[0] != batch_size:
                raise ValueError(f"Expected batched output to have the same batch size as the other batched outputs")
        unbatched_shape = outer_batched.shape[1:]
        if inner_unbatched.shape != unbatched_shape:
            raise ValueError(f"Expected inner unbatched shape to be {unbatched_shape}, but got {inner_unbatched.shape}")
        
        # if outer_batched not in outer_graph.node_table:
        #     raise ValueError(f"Expected outer batched variable to be in the outer graph")
        if inner_unbatched not in body_graph.node_table:
            raise ValueError(f"Expected inner unbatched variable to be in the body graph")

        outer.append(outer_batched)
        inner.append(inner_unbatched)
    return outer, inner, batch_size

def check_unbatched_inputs(unbatched_inputs:list[Variable], outer_graph, body_graph):
    for input in unbatched_inputs:
        # if input not in outer_graph.node_table:
        #     raise ValueError(f"Expected unbatched input to be in the outer graph")
        if input not in body_graph.node_table:
            raise ValueError(f"Expected unbatched input to be in the body graph")

def get_bloop_inputs(body_graph, outer_graph):
    unbatched_inputs = []
    for node in body_graph.node_table:
        if body_graph.in_degree(node) > 0:
            continue
        if node in outer_graph.node_table:
            unbatched_inputs.append(node)
    return unbatched_inputs

class BLoop(SubgraphOperation):

    def __init__(
            self,
            body,
            batched_inputs:list[tuple[Variable,Variable]],
            unbatched_inputs:list[Variable],
            batched_outputs:list[tuple[Variable,Variable]],
            name:str = 'bloop',
        ) -> None:
        from csdl_alpha.src.graph.graph import Graph
        import csdl_alpha as csdl
        current_recorder = csdl.get_current_recorder()
        # Add checks:
        if not isinstance(body, Graph):
            raise TypeError(f"Expected body to be a Graph, but got {get_type_string(body)}")
        
        batched_outer_in, unbatched_inner_in, batch_size_o = check_batched_variable_pairs(
            batched_inputs,
            current_recorder.active_graph,
            body,
        )
        batched_outer_out, unbatched_inner_out, batch_size_in = check_batched_variable_pairs(
            batched_outputs,
            current_recorder.active_graph,
            body,
        )
        check_unbatched_inputs(
            unbatched_inputs,
            current_recorder.active_graph,
            body,
        )

        # Check batch size consistency across all batched inputs and outputs
        if batch_size_o != batch_size_in:
            raise ValueError(f"Expected batch sizes to be the same, but got {batch_size_o} and {batch_size_in}")
        self.batch_size = batch_size_o

        
        # Store metadata
        self.name = name
        body.name = name
        self.outer_batched_inputs = batched_outer_in
        self.inner_unbatched_inputs = unbatched_inner_in
        self.unbatched_inputs = unbatched_inputs

        self.outer_batched_outputs = batched_outer_out
        self.inner_unbatched_outputs = unbatched_inner_out

        # Input variables
        all_inputs = self.outer_batched_inputs  + unbatched_inputs
        super().__init__(*all_inputs)

        # Output variables
        self.set_outputs(self.outer_batched_outputs)
        
        # Assign subgraph
        self.assign_subgraph(body)

    def compute_inline(self, *args):
        
        import numpy as np
        # pre_allocate batched outputs
        for batched_outer in self.outer_batched_outputs:
            batched_outer.value = np.zeros(batched_outer.shape)
        
        for i in range(self.batch_size):
            for inner_unbatched, outer_batched in zip(self.inner_unbatched_inputs, self.outer_batched_inputs):
                inner_unbatched.value = outer_batched.value[i]

            self.get_subgraph().execute_inline()

            for inner_unbatched, outer_batched in zip(self.inner_unbatched_outputs, self.outer_batched_outputs):
                outer_batched.value[i] = inner_unbatched.value

        if len(self.outer_batched_outputs) == 1:
            return self.outer_batched_outputs[0].value
        else:
            return tuple([output.value for output in self.outer_batched_outputs])
        
    def compute_jax(self, *jax_inputs):
        outer_batched_inputs = jax_inputs[:len(self.outer_batched_inputs)]
        unbatched_inputs = jax_inputs[len(self.outer_batched_inputs):]

        import jax
        from csdl_alpha.backends.jax.graph_to_jax import create_jax_function
        body_func = create_jax_function(
            graph = self.get_subgraph(),
            outputs = self.inner_unbatched_outputs,
            inputs = self.inner_unbatched_inputs + self.unbatched_inputs,
        )

        in_axes = [0]*len(outer_batched_inputs) + [None]*len(unbatched_inputs)
        batched_body_func = jax.vmap(body_func, in_axes=tuple(in_axes))
        batched_outputs = batched_body_func(*outer_batched_inputs, *unbatched_inputs)
        return tuple(batched_outputs)

    def evaluate_vjp(self, cotangents, *inputs_and_outputs):
        
        from csdl_alpha.src.operations.derivatives.reverse import vjp
        from csdl_alpha.src.graph.graph import Graph, _copy_to_current_graph
        from csdl_alpha.src.operations.loops.vloop_utils import (
            build_static_input_data,
            build_batched_input_data,
            build_batched_output_data,
            BatchedData,
            StaticInputData,
        )

        # Preprocessing:
        # organize static inputs
        static_inputs, remainder_static_inputs = build_static_input_data(
            self,
            cotangents,
        )

        # organize batched inputs
        batched_inputs, remainder_batched_inputs = build_batched_input_data(
            self,
            cotangents,
        )

        # organize batched outputs
        batched_outputs, remainder_batched_outputs = build_batched_output_data(
            self,
            cotangents,
        )

        # Build inner graph
        # Create adjoint Bloop:
        import csdl_alpha as csdl
        recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')
        original_body_graph:Graph = self.get_subgraph()
        outer_graph = recorder.active_graph
        recorder._enter_subgraph(
            name = original_body_graph.name+'_vjp',
            add_missing_variables=True,
            )
        new_body_graph:Graph = recorder.active_graph
        
        # Compute forward
        _copy_to_current_graph(original_body_graph, {})

        for batched_output in batched_outputs:
            batched_output.unbatched_cotangent = Variable(
                name = f'{batched_output.unbatched.name}_unb_seed',
                value = np.ones(batched_output.unbatched.shape),
            )
        # Compute vjp
        vjps = vjp(
            seeds = [(bd.unbatched, bd.unbatched_cotangent) for bd in batched_outputs],
            wrts = [bd.unbatched for bd in batched_inputs]+[si.external_body_input for si in static_inputs],
            graph = new_body_graph,
        )
        for bd in batched_inputs:
            bd.unbatched_cotangent = vjps[bd.unbatched]
        for si in static_inputs:
            si.internal_cotangent = vjps[si.external_body_input]
        
        # end inner graph
        recorder._exit_subgraph()

        # Compute vjp batched inputs:
        # - original batched inputs + remainders
        # - original output cotangents
        vjp_batched_inputs = [(bd.batched, bd.unbatched) for bd in batched_inputs]
        vjp_batched_inputs += remainder_batched_inputs
        vjp_batched_inputs += [(bd.batched_cotangent, bd.unbatched_cotangent) for bd in batched_outputs]

        # Compute vjp static inputs:
        # - original static inputs + remainders
        vjp_static_inputs = [si.external_body_input for si in static_inputs]
        vjp_static_inputs += remainder_static_inputs

        # Compute vjp batched outputs:
        # - original static input cotangents
        # - original batched input cotangents
        vjp_batched_outputs = [(bd.batched_cotangent, bd.unbatched_cotangent) for bd in batched_inputs if bd.unbatched_cotangent is not None]
        vjp_batched_outputs += [(si.external_cotangent, si.internal_cotangent) for si in static_inputs if si.internal_cotangent is not None]

        if len(vjp_batched_outputs) > 0:
            # build vloop
            bloop = BLoop(
                body=new_body_graph,
                batched_inputs=vjp_batched_inputs,
                unbatched_inputs=vjp_static_inputs,
                batched_outputs=vjp_batched_outputs,
                name = new_body_graph.name,
            )
            bloop.finalize_and_return_outputs()

        # Finally accumulate corangents:
        for bd in batched_inputs:
            cotangents.accumulate(bd.batched,bd.batched_cotangent)
        for si in static_inputs:
            cotangents.accumulate(si.external_body_input, csdl.sum(si.external_cotangent, axes = (0,)))
    