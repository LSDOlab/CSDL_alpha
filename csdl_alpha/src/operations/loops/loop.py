from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.operations.operation_subclasses import SubgraphOperation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.operations.set_get.loop_slice import _loop_slice as slice
from typing import Union

import numpy as np

class IterationVariable(Variable):
    def __init__(self, vals):
        super().__init__(value=vals[0])
        # self.shape = (1,)
        self.vals = vals
        self.name = 'iter'

class Loop(SubgraphOperation):

    def __init__(self, inputs, outputs, graph, vals, iter_vars, loop_vars, name = 'loop', parent = None) -> None:
        """
        Initialize a Loop object.

        Parameters
        ----------
        inputs : list
            List of input nodes for the loop.
        outputs : list
            List of output nodes for the loop.
        graph : Graph
            The graph representing one loop iteration.
        vals : list
            List of lists of values corresponding to each iteration variable.
        iter_vars : list
            List of iteration variables.
        loop_vars : tuple
            Tuple containing the input node in the graph, input for the first iteration, and input for subsequent iterations.
            List is length 2n - 0:n are the natural loop vars, and n:2n are loop vars for stacking 0:n
        """
        super().__init__(*inputs)
        self.name = name
        self.inputs = inputs
        self.num_inputs = len(inputs)
        self.outputs = outputs
        self.graph = graph
        self.vals = vals # list of list of values corresponding to iter_var
        self.iter_vars = iter_vars # list of iteration variables
        self.loop_vars = loop_vars # (input node in graph, input for first iter, input for subsiquent iters)
        self.has_reset = False
        self.loop_var_history = {loop_var:[] for loop_var in loop_vars}
        self.length = len(self.vals[0])
        self.parent = parent

        self.loop_var_lookup:dict[Variable:Variable] = {}
        self.num_loop_vars = len(loop_vars)
        for i in range(self.num_loop_vars):
            self.loop_var_lookup[loop_vars[i][2]] = self.outputs[-(self.num_loop_vars-i)]
        
        self._add_outputs_to_graph()
        self._add_to_graph()
        self.assign_subgraph(graph)

    def get_stacked(self, variable:Variable) -> Variable:
        """Gets the stacked input per iteration for a given loop outout with feedback

        Parameters
        ----------
        variable : Variable
            Loop output to get the stacked variable for

        Returns
        -------
        Variable
            Stacked variable corresponding to the loop output. Shape is (num_iterations, *variable.shape)
        """
        return self.loop_var_lookup[variable]


    def _add_outputs_to_graph(self):
        for output in self.outputs:
            self.recorder.active_graph.add_node(output)

    def compute_inline(self, *args):
        
        # If a derivative loop, we need to reset the intermediate variables
        if self.parent:
            old_var_values = {}
            for intermediate_var in self.parent.get_subgraph().node_table.keys():
                if isinstance(intermediate_var, Variable):
                    old_var_values[intermediate_var] = intermediate_var.value

        import numpy as np
        # clear loop var history
        for hist in self.loop_var_history.values():
            hist.clear()

        # run loop
        for i in range(self.length):
                
            # Set feedback variables and iteration variable values
            for loop_var in self.loop_vars:
                if i == 0:
                    loop_var[0].value = loop_var[1].value
                self.loop_var_history[loop_var].append(loop_var[0].value)
            for iter_var, val in zip(self.iter_vars, self.vals):
                iter_var.set_value(val[i])

            # output stacked variables
            for ii in range(self.num_loop_vars):
                self.outputs[-(self.num_loop_vars-ii)].value[i] = self.loop_vars[ii][0].value
                
            self.graph.execute_inline()

            # Update feedback
            for loop_var in self.loop_vars:
                loop_var[0].value = loop_var[2].value

        # If a derivative loop, we need to reset the intermediate variables
        if self.parent:
            for intermediate_var in old_var_values:
                intermediate_var.value = old_var_values[intermediate_var]

        return [output.value for output in self.outputs]

    def evaluate_vjp(self, cotangents, *inputs_and_outputs):
        inputs = inputs_and_outputs[:self.num_inputs]
        outputs = inputs_and_outputs[self.num_inputs:]
        debug = False
        if debug:
            print(f'loop VJP of {self.name}')

        # setup
        from csdl_alpha.src.operations.loops.utils import build_feedback_data, FeedBackData, build_reversed_iteration_variables, build_external_inputs_data, build_external_outputs_data

        import csdl_alpha as csdl
        from csdl_alpha.src.operations.derivative.reverse import vjp
        from csdl_alpha.src.graph.graph import _copy_to_current_graph
        outer_graph = csdl.get_current_recorder().active_graph

        # Preprocessing:
        # Organize feedbacks
        feedbacks:list[FeedBackData] = build_feedback_data(self, cotangents)
        
        # Organize iteration variables
        parent_iter_vars:list[Variable] = self.iter_vars

        # Organize external inputs (with cotangents)
        feedback_inputs:set[Variable] = {feedback.external_input for feedback in feedbacks}
        parent_external_inputs, remaining_external_inputs = build_external_inputs_data(self, feedback_inputs, cotangents)

        # Organize external outputs (with cotangents)
        feedback_outputs:set[Variable] = {feedback.body_external_output for feedback in feedbacks}
        feedback_outputs.update({feedback.input_stack for feedback in feedbacks})
        parent_external_outputs = build_external_outputs_data(self, feedback_outputs, cotangents)

        # Checks: Comment out later
        if debug:
            print('EXT OUTPUTS:')
            for output in parent_external_outputs:
                print(f'\t{output}:')
                print(f'\t\t{output.external_body_IO.name}')
                print(f'\t\t{output.external_input_cotangent.name}')
            print('EXT INPUTS:')
            for output in parent_external_inputs:
                print(f'\t{output}:')
                print(f'\t\t{output.external_body_IO.name}')
                print(f'\t\t{output.external_input_cotangent.name}')
            print('FEEDBACKS:')
            for feedback in feedbacks:
                print(f'\t{feedback}:')
                print(f'\t\tb/ext out {feedback.body_external_output.name} \t\t{feedback.body_external_output}')
                print(f'\t\text in    {feedback.external_input.name} \t\t{feedback.external_input}')
                print(f'\t\tbody in   {feedback.body_input.name} \t\t{feedback.body_input}')
                print(f'\t\text cot   {feedback.external_input_cotangent.name} \t\t{feedback.external_input_cotangent}')
                print(f'\t\tstacked   {feedback.input_stack.name} \t\t{feedback.input_stack}')

            for output in parent_external_outputs:
                # TODO: Find another way to check this
                # assert output.external_body_IO in outer_graph.node_table
                assert output.external_input_cotangent in outer_graph.node_table
            for input in parent_external_inputs:
                # TODO: Find another way to check this
                # assert input.external_body_IO in outer_graph.node_table
                assert input.external_input_cotangent in outer_graph.node_table
            for feedback in feedbacks:
                # TODO: Find another way to check this
                # assert feedback.body_external_output in outer_graph.node_table
                # assert feedback.external_input in outer_graph.node_table
                assert feedback.body_input in self.get_subgraph().node_table
                assert feedback.external_input_cotangent in outer_graph.node_table

        parent_loop_graph = self.get_subgraph()

        # ========================================================================================================================================
        # ====================== VJP LOOP BODY FUNCTION ==========================================================================================
        # =============================== START ==================================================================================================
        # ========================================================================================================================================

        vjp_external_inputs = []
        vjp_external_ouputs = []

        # Add parent inputs to the VJP loop that are not used for cotangents:
        for non_accumulating_input in remaining_external_inputs:
            vjp_external_inputs.append(non_accumulating_input)

        # Initialize body of VJP loop
        recorder = csdl.get_current_recorder()
        recorder._enter_subgraph(
            name = parent_loop_graph.name+'_vjp',
            add_missing_variables=True
            )        
        deriv_loop_graph = recorder.active_graph

        # Create new iteration variables
        rev_index, rev_orig = build_reversed_iteration_variables(parent_iter_vars)
        if debug:
            for parent_iter, rev_iter in zip(parent_iter_vars, rev_orig):
                print(parent_iter.vals, rev_iter.vals)
            print(rev_index.vals)

        # Connect new body inputs:
        # 1) new reversed iteration variables   --> set to -->   original iteration variables
        # 2) stacked feedback variable          --> set to -->   original feedback variable
        vjp_body_inputs_map:dict[Variable:Variable] = {}
        
        # 1)
        for parent_iter_var, rev_iter_var in zip(parent_iter_vars, rev_orig):
            vjp_body_inputs_map[parent_iter_var] = rev_iter_var
        # 2)
        for feedback in feedbacks:
            vjp_external_inputs.append(feedback.input_stack)
            vjp_body_inputs_map[feedback.body_input] = feedback.input_stack[rev_index]

        # Body forward evaluation
        _copy_to_current_graph(parent_loop_graph, vjp_body_inputs_map)
        # deriv_loop_graph.visualize('vjp_loop')

        # Compute vector jacobian products
        # of: feedback body external outputs, external outputs with cotangents
        # wrt: feedback body inputs, external inputs with cotangents
        
        # ofs:
        seeds = []
        for parent_external_output in parent_external_outputs:
            # For each feedback, we need to create a new cotangent body input
            # we connect these for feedback later 
            ext_output = parent_external_output.external_body_IO
            parent_external_output.body_input_cotangent = csdl.Variable(
                name = f'{ext_output.name}_tangent_body',
                shape = parent_external_output.external_input_cotangent.shape,
                value=parent_external_output.external_input_cotangent.value,
            )
            seeds.append((ext_output,parent_external_output.body_input_cotangent))

            # For feedbacks later
            vjp_external_inputs.append(parent_external_output.external_input_cotangent)

        for feedback in feedbacks:
            # For each feedback, we need to create a new cotangent body input
            # we connect these for feedback later 
            feedback.body_input_cotangent = csdl.Variable(
                name=f'{feedback.body_external_output.name}_tangent_body',
                shape=feedback.external_input_cotangent.shape,
                value=feedback.external_input_cotangent.value,
            )
            seeds.append((feedback.body_external_output, feedback.body_input_cotangent))

            # For feedbacks later
            vjp_external_inputs.append(feedback.external_input_cotangent)

        # wrts:
        wrts = [feedback.body_input for feedback in feedbacks] + [external_in.external_body_IO for external_in in parent_external_inputs]
        
        # Finally compute the vector jacobian products
        vjps = vjp(seeds, wrts, deriv_loop_graph)

        # Perform loop accumulation:
        for feedback in feedbacks:
            if cotangents.check(feedback.input_stack):
                vjp_external_inputs.append(cotangents[feedback.input_stack])
                accumulated = vjps[feedback.body_input] + cotangents[feedback.input_stack][rev_index]
            else:
                accumulated = vjps[feedback.body_input]

            feedback.out_cotangent = accumulated
            vjp_external_ouputs.append(feedback.out_cotangent)
        for parent_external_input in parent_external_inputs:
            parent_external_input.body_input_cotangent = csdl.Variable(
                name = f'{parent_external_input.external_body_IO.name}_tangent_body',
                shape = parent_external_input.external_body_IO.shape,
                value = np.zeros(parent_external_input.external_body_IO.shape)
            )
            if vjps[parent_external_input.external_body_IO] is None:
                parent_external_input.out_cotangent = parent_external_input.body_input_cotangent
            else:
                # This is a feedback right here --> These variables are stacked --> This feedback could be of an ALREADY stacked variable
                # Therefore we could be DOUBLE/TRIPLE/etc stacking with every higher order derivative. --> Figure out a way to avoid this
                # TODO: Can we get rid of this stacking for a +=?
                parent_external_input.out_cotangent = vjps[parent_external_input.external_body_IO] + parent_external_input.body_input_cotangent
                parent_external_input.out_cotangent.add_name(f'{parent_external_input.external_body_IO.name}_out_cotangent')
            vjp_external_ouputs.append(parent_external_input.out_cotangent)
            vjp_external_inputs.append(parent_external_input.external_input_cotangent)
            vjp_external_inputs.append(parent_external_input.external_body_IO)
        
        for parent_external_output in parent_external_outputs:
            parent_external_output.out_cotangent = parent_external_output.body_input_cotangent*0.0
            parent_external_output.out_cotangent.add_name(f'{parent_external_output.external_body_IO.name}_zeroed')
            vjp_external_ouputs.append(parent_external_output.out_cotangent)

        # deriv_loop_graph.visualize('vjp_loop2')
        recorder._exit_subgraph()

        # ========================================================================================================================================
        # ====================== VJP LOOP BODY FUNCTION ==========================================================================================
        # =============================== END ====================================================================================================
        # ========================================================================================================================================

        # Assign feedbacks
        vjp_loop_vars = []
        stacked_outputs = []
        for feedback_data in feedbacks + parent_external_inputs + parent_external_outputs:
            vjp_loop_vars.append((feedback_data.body_input_cotangent, feedback_data.external_input_cotangent, feedback_data.out_cotangent))

            stacked_outputs.append(build_stacked_variable(feedback_data.body_input_cotangent, self.length))
            
            # Checks:
            if debug:
                print(feedback_data)
                print('\t', feedback_data.body_input_cotangent)
                print('\t', feedback_data.external_input_cotangent)
                print('\t', feedback_data.out_cotangent)

                assert feedback_data.body_input_cotangent in deriv_loop_graph.node_table
                assert feedback_data.external_input_cotangent in outer_graph.node_table
                assert feedback_data.out_cotangent in deriv_loop_graph.node_table
        
        # Build the loop operation:
        vjp_loop_op = Loop(
            inputs = vjp_external_inputs,
            outputs = vjp_external_ouputs+stacked_outputs,
            graph = deriv_loop_graph,
            vals = [rev_index.vals] + [rev_iter.vals for rev_iter in rev_orig],
            iter_vars = [rev_index] + rev_orig,
            loop_vars = vjp_loop_vars,
            name = f'vjp_{self.name}',
            parent = self,
        )
        vjp_loop_op.finalize_and_return_outputs()

        # outer_graph.visualize('outer_post_vjp')

        # Finally accumulate the cotangents ...
        for feedback in feedbacks:
            if cotangents.check(feedback.external_input):
                cotangents.accumulate(feedback.external_input, feedback.out_cotangent)
        for parent_external_input in parent_external_inputs:
            cotangents.accumulate(parent_external_input.external_body_IO, parent_external_input.out_cotangent)

class frange():
    def __init__(self, arg1:int=None, arg2:int=None, increment:int=1, *, vals:Union[list[int],tuple[list[int]]] = None):
            """Initialize the Loop object.

            Parameters
            ----------
            arg1 : int, optional
                The lower bound of the loop. If `arg2` is not provided, `arg1` represents the upper bound of the loop.
            arg2 : int, optional
                The upper bound of the loop. If provided, `arg1` represents the lower bound of the loop.
            increment : int, optional
                The increment value for each iteration of the loop. By default, it is set to 1.
            vals : list[int] or tuple[list[int]], optional
                A list or tuple of lists of values to iterate over.

            Raises
            ------
            ValueError
                If the lower bound of the loop is greater than the upper bound.
            ValueError
                If any value in the `vals` list or tuple of lists is not an integer.
            """

            if arg2 is None:
                if arg1 is None:
                    if vals is None:
                        raise ValueError(f'No arguments provided for the for loop')
                else:
                    lower = 0
                    upper = arg1
            else:
                lower = arg1
                upper = arg2

            # process runtime iterations
            if vals is None:
                if upper < lower:
                    raise ValueError(f'The lower bound of the for loop, {lower}, is above the upper bound of the for loop, {upper}')
                self.vals = [list(range(lower, upper, increment))]
            elif isinstance(vals, list):
                if not all(isinstance(val, int) for val in vals):
                    raise ValueError(f'All values in the list of values must be integers')
                self.vals = [vals]
            elif isinstance(vals, tuple):
                for vals_list in vals:
                    if not all(isinstance(val, int) for val in vals_list):
                        raise ValueError(f'All values in the list of values must be integers')
                self.vals = list(vals)

            self.curr_index = 0
            self.max_index = 2

            # enter new graph
            # TODO: Enter new subgraph only when the actual iteration begins
            # TODO: Add a check to make sure they only use this frange object once!
            from csdl_alpha.api import manager
            self._recorder = manager.active_recorder
            self._recorder._enter_subgraph(add_missing_variables=True, name = 'loop')
            self._graph = self._recorder.active_graph
            # self._graph_node = self._recorder.active_graph_node

            # initialize iteration variable:
            self.iteration_variables = []
            for vals in self.vals:
                self.iteration_variables.append(IterationVariable(vals))


    def get_ops_and_shapes(self, graph=None):
        ops = []
        shapes = []
        if graph is None:
            graph = self._recorder.active_graph
        for node in graph.node_table.keys():
            if isinstance(node, Operation):
                ops.append(type(node))
            elif isinstance(node, Variable):
                shapes.append(node.shape)
        return ops, shapes

    def post_iteration_one(self):
        # self._graph.visualize('graph_loop_iter_1')
        self.iter1_inputs = [] # list of inputs to the first iteration
        self.iter1_outputs = [] # list of outputs to the first iteration
        # NOTE: variables that are created inside the loop but not used in the loop aren't going to show up in either of these lists, but that *should* be okay?
        ops = []
        self.iter1_non_inputs = set() # list of all other variables in first iteration (will be removed later)
        for node in self._graph.node_table.keys():
            if isinstance(node, Operation):
                ops.append(node)
                for input in node.inputs:
                    if self._graph.in_degree(input)==0:
                        self.iter1_inputs.append(input)
                for output in node.outputs:
                    self.iter1_outputs.append(output)
            else:
                self.iter1_non_inputs.add(node)

        for input in self.iter1_inputs:
            self.iter1_non_inputs.discard(input)

        # don't want iteration variable to be removed, even if it's not used
        for iteration_variable in self.iteration_variables:
            self.iter1_non_inputs.discard(iteration_variable)

        # deleting the operations so we cana find inputs to the second iteration in the same way
        self._graph._delete_nodes(ops)

    def post_iteration_two(self):
        # self._graph.visualize('graph_loop_iter_2')
        self.iter2_inputs = [] # list of inputs to the second iteration (same order as first)
        self.iter2_outputs = [] # list of outputs to the second iteration (same order as first)
        for node in self._graph.node_table.keys():
            if isinstance(node, Operation):
                for input in node.inputs:
                    if self._graph.in_degree(input)==0:
                        self.iter2_inputs.append(input)
                for output in node.outputs:
                    self.iter2_outputs.append(output)

        # any input that's changed represents an internal loop, so we need to replace it with a special variable
        loop_vars = []
        strike_set = set() # set of inputs that are only used in the first iteration (feedback)
        for input1, input2 in zip(self.iter1_inputs, self.iter2_inputs):
            if not input1 is input2: 
                if input2 in self.iter1_outputs:
                        # we want to go from input2 to the corresponding output of the 2nd iteration
                        output2 = self.iter2_outputs[self.iter1_outputs.index(input2)] # TODO: make this less bad
                        loop_var = (input2, input1, output2) # (input node in graph, input for first iter, input for subsiquent iters)
                        
                        # OLD:
                        # if input1 in self._graph.node_table.keys():
                            # self._graph._delete_nodes([input1])
                        
                        # NEW:
                        if input1 in self._graph.node_table.keys():
                            if not (input1 in self.iter2_inputs):
                                self._graph._delete_nodes([input1])

                        self.iter1_non_inputs.discard(input2)
                        
                        # TODO: this is a bit of a hack, but it works for now
                        if loop_var in loop_vars:
                            continue
                        loop_vars.append(loop_var)
                else:
                    # this implies input 1 and input 2 are both made in the loop, so we can just keep input 2
                    pass
        # remove any inputs that are no longer used
        self._graph._delete_nodes(list(strike_set))

        # delete any remnanats of the first iteration
        self._graph._delete_nodes(self.iter1_non_inputs)
        # self._graph.visualize('graph_loop_iter_3')

        external_inputs = self._graph.inputs
        # non_feedback_inputs = external_inputs - strike_set # external inputs that are used for things other than feedback (and maybe feedback too)
        # Stop the graph
        # self._graph.visualize('graph_loop_final')
        self._recorder._exit_subgraph()

        for loop_var in loop_vars:
            # stack_output = Variable(name = f'stack_out_{loop_var[1].name}', shape=(len(self.vals[0]),) + loop_var[0].shape, value=0)
            stack_output = build_stacked_variable(loop_var[0], len(self.vals[0]))
            self.iter2_outputs.append(stack_output)
            
        # add the loop operation to the graph
        #NOTE: this only exposes outputs of operations, not variables created within the loop
        self.op = Loop(
            external_inputs, 
            self.iter2_outputs, 
            self._graph, 
            self.vals, 
            self.iteration_variables, 
            loop_vars
            )

    def _check_ops_and_shapes(self, ops, shapes):
        if ops != self.ops:
            raise ValueError(f'Operations changed between iterations')
        if shapes != self.shapes:
            raise ValueError(f'Shapes changed between iterations')


    def __next__(self):
        final = False
        # no processing for zeroith iteration
        if self.curr_index==0:
            self.in_loop = self._recorder._in_loop
            self._recorder._in_loop = True
        # first iteration - figure out inputs
        if self.curr_index==1:
            self.post_iteration_one()
        # second iteration - check feedback
        elif self.curr_index == 2:
            final = True
            self.post_iteration_two()

        if final:
            # print(f'loop vars are {self.op.loop_vars}')
            self._recorder._in_loop = self.in_loop
            if self._recorder.inline and not self.in_loop:
                # print('running_loop_inline')
                self.op.compute_inline()
            raise StopIteration

        self.curr_index+=1
        if len(self.iteration_variables) == 1:
            return self.iteration_variables[0]
        return tuple(self.iteration_variables)
        
    def __iter__(self):
        return self

def build_stacked_variable(
        feedback_var:Variable,
        num_iter:int,
    ):
    stack_output = Variable(name = f'stack_out_{feedback_var.name}', shape=(num_iter,) + feedback_var.shape, value=0)
    return stack_output

def add_compute_inline_reset(deriv_loop:Loop, old_loop:Loop):
    
    def _compute_inline_reset(self:Loop, *args):

        old_var_values = {}
        for intermediate_var in old_loop.get_subgraph().node_table.keys():
            if isinstance(intermediate_var, Variable):
                old_var_values[intermediate_var] = intermediate_var.value

        self.compute_inline(*args)
        
        for intermediate_var in old_loop.get_subgraph().node_table.keys():
            if isinstance(intermediate_var, Variable):
                intermediate_var.value = old_var_values[intermediate_var]

    deriv_loop.compute_inline = _compute_inline_reset

if __name__ == '__main__':
    import csdl_alpha as csdl
    from csdl_alpha.src.operations.add import Add
    import numpy as np
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    dim = 10
    b = csdl.Variable(value=np.zeros((dim,dim)), name='b')
    c = csdl.Variable(value=np.ones((dim, dim)), name='c')

    for i in frange(dim):
        for j in frange(dim):
            for k in frange(dim):
                b = b.set(csdl.slice[i, j], c[i,j])
                b = b*2

    b_np = np.zeros((dim,dim))
    c_np = np.ones((dim,dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                b_np[i,j] = c_np[i,j]
                b_np = b_np*2


    print(b.value-b_np)
    exit()
    # print('==============')
    # print(c.value)
    # print(c_np)

    # print(recorder.active_graph)

    top_graph_node = recorder.active_graph_node
    outer_loop_graph_node = top_graph_node.children[0]

    top_graph = top_graph_node.value
    outer_loop_graph = outer_loop_graph_node.value
    inner_loop_graph = outer_loop_graph_node.children[0].value

    top_graph.visualize('top_graph')
    outer_loop_graph.visualize('outer_loop_graph')
    inner_loop_graph.visualize('inner_loop_graph')


    # for i in vrange(0, 10, check=True):
    #     print(f'begin outer iteration {k}')
    #     k += 1
    #     l = 0
    #     for j in vrange(0, 10, check=True):
    #         print(f'inner iteration {l}')
    #         l += 1
    #         d = i*2
    #         e = i*j
    #         b2 = a + b
    #         c = a*2
    #     print(f'end outer iteration {k}')

    # top_graph_node = recorder.active_graph_node
    # outer_loop_graph_node = top_graph_node.children[0]
    # inner_loop_graph_node = outer_loop_graph_node.children[0]

    # top_graph = top_graph_node.value
    # outer_loop_graph = outer_loop_graph_node.value
    # inner_loop_graph = inner_loop_graph_node.value

    # top_graph.visualize('top_graph')
    # outer_loop_graph.visualize('outer_loop_graph')
    # inner_loop_graph.visualize('inner_loop_graph')

    # print(d.value) # should be 18
    # print(e.value) # should be 81
    # print(b2.value) # should be 5
    # print(c.value) # should be 4
    # recorder.active_graph.visualize('outer_graph')
    # recorder.stop()