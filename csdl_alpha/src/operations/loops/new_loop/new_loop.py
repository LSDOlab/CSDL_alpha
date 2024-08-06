from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.operations.operation_subclasses import SubgraphOperation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.graph import _copy_to_current_graph
from csdl_alpha.src.operations.loops.new_loop.loop_builder import LoopBuilder, IterationVariable
from csdl_alpha.src.operations.loops.new_loop.utils import (
    build_loop_deriv_data,
    FeedbackDeriv,
    ParentIODeriv,
    MetaOutputDeriv,
    reverse_iteration_values,
    add_to_seed_dict,
)

from typing import Union
import numpy as np
import csdl_alpha as csdl

class NewLoop(SubgraphOperation):

    def __init__(
            self,
            loop_builder:LoopBuilder,
            parent:'NewLoop' = None,
            name:str = None,
            stack_all:bool = False,
        ):

        self.loop_builder:LoopBuilder = loop_builder
        self.length:int = self.loop_builder.length
        self.parent = parent
        self.stack_all:bool = stack_all
        if name is None:
            self.name = 'new_loop'
        else:
            self.name = name

        # Process inputs
        # Two types of inputs
        # 1. "Standard" inputs
        # 2. "Feedback" inputs
        self.operation_inputs = []
        for input_var in self.loop_builder.inputs:
            self.operation_inputs.append(input_var)
        for feedback in self.loop_builder.feedbacks._int_input_to_feedback.values():
            if feedback.external_input not in self.loop_builder.inputs:
                self.operation_inputs.append(feedback.external_input)

        # Process outputs
        # Three types of outputs:
        # 1. "Standard" outputs
        # 2. "Accrued" outputs
        # 3. "Stacked" outputs
        operation_outputs = []
        for output in self.loop_builder.outputs:
            recorder = csdl.get_current_recorder()
            if output in recorder.active_graph.node_table:
                raise ValueError(f"Output {output} already exists in graph.")
            else:
                recorder._add_node(output)
            operation_outputs.append(output)
        for output in self.loop_builder.accrued.values():
            operation_outputs.append(output)
        for output in self.loop_builder.stacked.values():
            operation_outputs.append(output)
        
        super().__init__(*self.operation_inputs)
        self.set_outputs(operation_outputs)
        self.assign_subgraph(loop_builder.loop_graph)

    def compute_inline(self, *args):

        # If a derivative loop, we need to reset the intermediate variables
        if self.parent:
            old_var_values = {}
            for intermediate_var in self.parent.get_subgraph().node_table.keys():
                if isinstance(intermediate_var, Variable):
                    old_var_values[intermediate_var] = intermediate_var.value

        # Preprocess the stack and accrue variables
        for accrue_target, accrued_var in self.loop_builder.accrued.items():
            accrued_var.value = np.zeros(accrued_var.shape)
        for stack_target, stacked_var in self.loop_builder.stacked.items():
            stacked_var.value = np.zeros(stacked_var.shape)

        # Perform the "real" loop here
        for i in range(self.length):
            # set the iteration variable values for current iteration
            for iter_var, value in self.loop_builder.iters.items():
                iter_var.value = value[i]

            # set the feedback variables for first iteration
            if i == 0:
                for feedback in self.loop_builder.feedbacks._int_input_to_feedback.values():
                    feedback.internal_input.value = feedback.external_input.value

            self.get_subgraph().execute_inline()

            # update accrued variables
            for accrue_target, accrued_var in self.loop_builder.accrued.items():
                accrued_var.value = accrued_var.value + accrue_target.value

            # update stacked variables
            for stack_target, stacked_var in self.loop_builder.stacked.items():
                stacked_var.value[i] = stack_target.value

            # update feedback variables
            if i < self.length - 1:
                for feedback in self.loop_builder.feedbacks._int_input_to_feedback.values():
                    feedback.internal_input.value = feedback.output.value

        # If a derivative loop, we need to reset the intermediate variables
        if self.parent:
            for intermediate_var in old_var_values:
                intermediate_var.value = old_var_values[intermediate_var]

        if len(self.outputs) == 1:
            return self.outputs[0].value
        else:
            return [output.value for output in self.outputs]

    def evaluate_jax(self, inputs:dict[csdl.Variable], outputs:dict[csdl.Variable]):
        import jax.numpy as jnp
        from csdl_alpha.backends.jax.graph_to_jax import create_jax_function
        import jax.lax as lax
        
        in2feed = self.loop_builder.feedbacks._int_input_to_feedback

        # build graph function:

        # meta outputs first
        # Remember, loop_builder.outputs contains all :
        #  - feedback outputs
        #  - accrued output targets
        #  - stacked output targets
        #  - standard outputs
        ordered_accrued_targets = [accrued_target for accrued_target in self.loop_builder.accrued.keys()]
        ordered_stacked_targets = [stacked_target for stacked_target in self.loop_builder.stacked.keys()]
        ordered_std_outputs = [std_output for std_output in self.loop_builder.outputs]
        std_outputs_indices = {std_output:i for i,std_output in enumerate(ordered_std_outputs)}

        # Now we need to build all possible outputs for the graph function which include the above outputs
        ordered_all_outputs = ordered_accrued_targets + ordered_stacked_targets + ordered_std_outputs
        all_output_indices = {std_output:i for i,std_output in enumerate(ordered_all_outputs)}
        ordered_std_inputs = list(self.loop_builder.inputs.keys())
        ordered_feedback_inputs = list(in2feed.keys())
        ordered_iter_vars = list(self.loop_builder.iters.keys())
        graph_fn = create_jax_function(
            self.get_subgraph(),
            ordered_all_outputs,
            ordered_std_inputs+ordered_feedback_inputs+ordered_iter_vars,
        )
        def loop_body(carry, x):
            # carried variables are going to be the feedback variables, outputs, and accrued variables
            # x is the iteration variable

            # set the iteration variable values for current iteration
            graph_fn_inputs = [inputs[body_input] for body_input in ordered_std_inputs]
            graph_fn_inputs += [carry[std_outputs_indices[in2feed[int_in].output]] for int_in in ordered_feedback_inputs]
            graph_fn_inputs += [x[i] for i,iter_var in enumerate(ordered_iter_vars)]

            # call the graph function
            graph_fn_outputs = graph_fn(*graph_fn_inputs)

            # update accrued variables
            accrued_outputs = []
            for accrue_ind, accrue_target in enumerate(ordered_accrued_targets):
                accrued_outputs.append(carry[accrue_ind+len(ordered_std_outputs)] + graph_fn_outputs[all_output_indices[accrue_target]])

            # update stacked variables
            stacked_outputs = [graph_fn_outputs[all_output_indices[stacked_target]] for stacked_target in ordered_stacked_targets]
            # print('len carry', len(carry), [v.size for v in graph_fn_outputs], [v.size for v in accrued_outputs])
            carry_out = []
            for output in ordered_std_outputs:
                jax_var = graph_fn_outputs[all_output_indices[output]]
                if jax_var.shape != output.shape:
                    jax_var = jax_var.reshape(output.shape)
                carry_out.append(jax_var)
            return carry_out+accrued_outputs, stacked_outputs

        # //////////////////// Set loop initial conditions: \\\\\\\\\\\\\\\\\\\\\
        iter_var_list = []
        for i in range(self.length):
            iter_var_list.append([iter_vals[i] for iter_var, iter_vals in self.loop_builder.iters.items()])
        iter_var_array = jnp.array(iter_var_list, dtype=jnp.float64)

        # non-accrued outputs first
        carry = [jnp.zeros(output.shape) for output in ordered_std_outputs]
        for feedback in self.loop_builder.feedbacks._int_input_to_feedback.values():
            ind = std_outputs_indices[feedback.output]
            carry[ind] = inputs[feedback.external_input]
        # accrued outputs
        carry += [jnp.zeros(output.shape) for output in ordered_accrued_targets]
        # \\\\\\\\\\\\\\\\\\\\\ Set loop initial conditions: ////////////////////

        # Actually run the loop now
        carry, stack = lax.scan(loop_body, carry, iter_var_array) 

        # Finally, process the outputs
        # standard outputs:
        for i, output in enumerate(ordered_std_outputs):
            outputs[output] = carry[i]
        
        # stacked outputs
        for i, output in enumerate(ordered_stacked_targets):
            stacked_var = self.loop_builder.stacked[output]
            outputs[stacked_var] = stack[i]

        for i, output in enumerate(ordered_accrued_targets):
            accrued_var = self.loop_builder.accrued[output]
            outputs[accrued_var] = carry[i+len(ordered_std_outputs)]

    # def prep_vjp(self):
    #     """
    #     Prepare the nonlinear solver for reverse mode differentiation.
    #     """
    #     import csdl_alpha as csdl
    #     if not self.stack_all:
    #         recorder = csdl.get_current_recorder()
    #         recorder._enter_subgraph(graph = self.get_subgraph())
            
    #         node_table = list(self.get_subgraph().node_table.keys())
    #         for node in node_table:
    #             if isinstance(node, Operation):
    #                 node.prep_vjp()
            
    #         recorder._exit_subgraph()
    #     else:
    #         pass

    def evaluate_vjp(self, cotangents, *inputs_and_outputs):
        inputs = inputs_and_outputs[:self.num_inputs]
        outputs = inputs_and_outputs[self.num_inputs:]
        debug = False
        if debug:
            print(f'loop VJP of {self.name}')

        # Preprocessing
        # - Organize all feedbacks
        # - Organize all iteration variables
        # - Outputs:
        # -- Organize all accrued outputs
        # -- Organize all stacked outputs
        # -- Organize all standard outputs
        # - Inputs:
        # -- Organize all standard inputs

        processed_vars = build_loop_deriv_data(self, cotangents)
        feedback_data:dict[Variable,FeedbackDeriv] = processed_vars[0]
        parent_iter_vars:dict[Variable] = processed_vars[1]
        parent_external_inputs:dict[Variable,ParentIODeriv] = processed_vars[2]
        parent_external_outputs:dict[Variable,ParentIODeriv] = processed_vars[3]
        parent_accrued_outputs:dict[Variable,MetaOutputDeriv] = processed_vars[4]
        parent_stacked_outputs:dict[Variable,MetaOutputDeriv] = processed_vars[5]

        # Build reversed iteration variables
        reversed_original_iteration_values:list[list[int]] = reverse_iteration_values(parent_iter_vars)
        reversed_indexing = [list(reversed(range(self.length)))]
        one_then_zeros = [[1] + [0]*(self.length-1)]
        new_iter = reversed_indexing+one_then_zeros+reversed_original_iteration_values

        #  //////////////////////////////////                      \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # ////////////////////////////////// Create derivative loop \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        from csdl_alpha.src.operations.derivatives.reverse import vjp
        with csdl.experimental.enter_loop(new_iter) as vjp_loop_builder:
            # Get new indices
            iter_vars = vjp_loop_builder.get_loop_indices()
            rev_index = iter_vars[0]
            one_if_first_iter = iter_vars[1]
            rev_orig = iter_vars[2:]

            rev_index.add_name('reversed iter')
            one_if_first_iter.add_name('one_zero_iter')
            # Process cotangents of outputs:

            rev_loop_graph = vjp_loop_builder.loop_graph
            if not self.stack_all:
                # If rebuild is True, we need to rebuild the body loop
                vjp_body_inputs_map:dict[Variable:Variable] = {}
        
                # 1)
                for parent_iter_var, rev_iter_var in zip(parent_iter_vars.keys(), rev_orig):
                    vjp_body_inputs_map[parent_iter_var] = rev_iter_var
                    assert isinstance(parent_iter_var, IterationVariable)
                    assert isinstance(rev_iter_var, IterationVariable)
                # 2)
                for int_input, feedback_deriv in feedback_data.items():
                    vjp_body_inputs_map[int_input] = feedback_deriv.stacked_internal_input[rev_index]

                # Insert body forward evaluation into derivative graph so I can take derivatives of it
                _copy_to_current_graph(
                    self.get_subgraph(),
                    vjp_body_inputs_map,
                    add_to_graph_inputs = True)
            else:
                # raise NotImplementedError('stack_all not implemented yet')
                stacked_inter_map:dict[Variable:Variable] = {}
                all_intermediate_vars = []
                for node in self.loop_builder.loop_graph.node_table.keys():
                    if not isinstance(node,Operation):
                        all_intermediate_vars.append(node)
                        if isinstance(node,IterationVariable):
                            continue
                        if node in self.loop_builder.inputs:
                            continue
                        if node not in self.loop_builder.stacked:
                            raise KeyError(f'INTERNAL ERROR: Node {node.info()} not in stacked outputs')
                        stacked_inter_map[node] = self.loop_builder.stacked[node][rev_index]
                for parent_iter_var, rev_iter_var in zip(parent_iter_vars.keys(), rev_orig):
                    stacked_inter_map[parent_iter_var] = rev_iter_var

                # Insert body forward evaluation VARIABLES into derivative graph
                _copy_to_current_graph(
                    self.get_subgraph(),
                    stacked_inter_map,
                    subgraph_nodes = all_intermediate_vars,
                    add_to_graph_inputs = True,
                )

                # loop_graph.visualize()
                # raise ValueError('stack_all not implemented yet')

            # Now compute the VJPs
            # Create seeds:
            seeds:dict[Variable,Variable] = {}
            # Standard output seeds
            for std_output_deriv in parent_external_outputs.values():
                ext_output_seed = (std_output_deriv.external_input_cotangent*one_if_first_iter)
                if ext_output_seed.shape != std_output_deriv.external_body_IO.shape:
                    ext_output_seed = ext_output_seed.reshape(std_output_deriv.external_body_IO.shape)
                add_to_seed_dict(seeds, std_output_deriv.external_body_IO, ext_output_seed)
            # Accrue output seeds
            for accrue_deriv in parent_accrued_outputs.values():
                add_to_seed_dict(seeds, accrue_deriv.internal_target, accrue_deriv.external_output_cotangent)
            # Stacked output seeds
            for stacked_deriv in parent_stacked_outputs.values():
                add_to_seed_dict(seeds, stacked_deriv.internal_target, stacked_deriv.external_output_cotangent[rev_index])
            # Feedback seeds
            for feedback_deriv in feedback_data.values():
                feedback_cot_x = vjp_loop_builder.initialize_feedback(feedback_deriv.external_input_cotangent)
                feedback_deriv.body_input_cotangent = feedback_cot_x
                add_to_seed_dict(seeds, feedback_deriv.parent_fb.output, feedback_cot_x)

            # create wrts:
            wrts = [feedback.parent_fb.internal_input for feedback in feedback_data.values()]
            wrts += [external_in.external_body_IO for external_in in parent_external_inputs.values()]
            
            # Finally compute the vector jacobian products
            vjps = vjp([(var,seed) for var,seed in seeds.items()], wrts, self.loop_builder.loop_graph)

            # Perform the accumulation procedures
            for feedback_deriv in feedback_data.values():
                feedback_deriv.out_cotangent = vjps[feedback_deriv.parent_fb.internal_input]
                vjp_loop_builder.finalize_feedback(
                    feedback_deriv.body_input_cotangent, feedback_deriv.out_cotangent)
            for external_in_deriv in parent_external_inputs.values():
                external_in_deriv.out_cotangent = vjps[external_in_deriv.external_body_IO]
        
        # Build the loop operation
        for external_in_deriv in parent_external_inputs.values():
            if external_in_deriv.out_cotangent is not None:
                external_in_deriv.out_cotangent = vjp_loop_builder.add_pure_accrue(external_in_deriv.out_cotangent)
            else:
                external_in_deriv.out_cotangent = csdl.Variable(value = np.zeros(external_in_deriv.external_body_IO.shape))
        for feedback_deriv in feedback_data.values():
            feedback_deriv.out_cotangent = vjp_loop_builder.add_output(feedback_deriv.out_cotangent)
        loop_op = vjp_loop_builder.finalize(
            add_all_outputs=False,
            name = f'vjp_of_loop_{self.name}',
            parent = self,
        )

        # post-processing of cotangents
        for external_in_deriv in parent_external_inputs.values():
            cotangents.accumulate(external_in_deriv.external_body_IO, external_in_deriv.out_cotangent)
        for feedback_deriv in feedback_data.values():
            fb_external_input = feedback_deriv.parent_fb.external_input
            # if fb_external_input not in parent_external_inputs:
            if cotangents.check(fb_external_input):
                cotangents.accumulate(fb_external_input, feedback_deriv.out_cotangent)