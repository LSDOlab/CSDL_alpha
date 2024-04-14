from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.graph.variable import Variable

class IterationVariable(Variable):
    def __init__(self, vals):
        super().__init__(value=vals[0])
        # self.shape = (1,)
        self.vals = vals
        self.name = 'iter'

# class LoopVariable(Variable):
#     def __init__(self, var1, var2):
#         super().__init__(shape=(var1.shape)) #NOTE: idk if we even need to do this
#         self.var1 = var1
#         self.var2 = var2
#         self.value = var1.value
#         self.latched = False
#         self.name = 'loop_var'

#     # def set_value(self, value):
#     #     if self.latched:
#     #         self.var2.set_value(value)
#     #     else:
#     #         self.var1.set_value(value)
#     #     super().set_value(value)

#     def reset(self):
#         # print('loop var reset')
#         self.latched = False
#         self.value = self.var1.value

#     def update_value(self):
#         if self.latched:
#             self.value = self.var2.value
#         else:
#             self.value = self.var1.value
#             self.latched = True
#         # print(f'Loop var updating to {self.value}')
        

class Loop(Operation):

    def __init__(self, inputs, outputs, graph, vals, iter_var, loop_vars) -> None:
        super().__init__()
        self.name = 'loop'
        self.inputs = inputs
        self.num_inputs = len(inputs)
        self.outputs = outputs
        self.graph = graph
        self.vals = vals
        self.iter_var = iter_var
        self.loop_vars = loop_vars # (input node in graph, input for first iter, input for subsiquent iters)
        self.has_reset = False
        self._add_outputs_to_graph()
        self._add_to_graph()

    def _add_outputs_to_graph(self):
        for output in self.outputs:
            self.recorder.active_graph.add_node(output)

    def compute_inline(self, *args):
        for i in range(len(self.vals)):
            if i == 0:
                for loop_var in self.loop_vars:
                    loop_var[0].value = loop_var[1].value
            self.iter_var.set_value(self.vals[i])
            self.graph.execute_inline()
            for loop_var in self.loop_vars:
                loop_var[0].value = loop_var[2].value
        return [output.value for output in self.outputs]


class frange():
    def __init__(self, arg1:int=None, arg2:int=None, increment:int=1, *, vals:list[int] = None):
            """Initialize the Loop object.

            Parameters
            ----------
            arg1 : int, optional
                The lower bound of the loop. If `arg2` is not provided, `arg1` represents the upper bound of the loop.
            arg2 : int, optional
                The upper bound of the loop. If provided, `arg1` represents the lower bound of the loop.
            increment : int, optional
                The increment value for each iteration of the loop. By default, it is set to 1.
            vals : list[int], optional
                A list of values to iterate over. If provided, the loop will iterate over the values in the list instead of using the range defined by `arg1` and `arg2`.

            Raises
            ------
            ValueError
                If the lower bound of the loop is greater than the upper bound.
            ValueError
                If any value in the `vals` list is not an integer.
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
                self.vals = list(range(lower, upper, increment))
            else:
                if not all(isinstance(val, int) for val in vals):
                    raise ValueError(f'All values in the list of values must be integers')
                self.vals = vals

            self.curr_index = 0
            self.max_index = 2

            # enter new graph
            from csdl_alpha.api import manager
            self._recorder = manager.active_recorder
            self._recorder._enter_subgraph(add_missing_variables=True)
            self._graph = self._recorder.active_graph
            self._graph_node = self._recorder.active_graph_node

            # initialize iteration variable:
            self.iteration_variable = IterationVariable(self.vals)

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
        self._graph.visualize('graph_loop_iter_1')
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
        self.iter1_non_inputs.discard(self.iteration_variable)

        # deleting the operations so we cana find inputs to the second iteration in the same way
        self._graph._delete_nodes(ops)

    def post_iteration_two(self):
        self._graph.visualize('graph_loop_iter_2')
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
                        loop_vars.append(loop_var)
                        self._graph._delete_nodes([input1])
                        self.iter1_non_inputs.remove(input2)
                else:
                    # this implies input 1 and input 2 are both made in the loop, so we can just keep input 2
                    self._graph._delete_nodes([input1])
                    pass
        # remove any inputs that are no longer used
        self._graph._delete_nodes(list(strike_set))

        # delete any remnanats of the first iteration
        self._graph._delete_nodes(self.iter1_non_inputs)

        external_inputs = self._graph.inputs
        # non_feedback_inputs = external_inputs - strike_set # external inputs that are used for things other than feedback (and maybe feedback too)

        # Stop the graph
        # self._graph.visualize('graph_loop_final')
        self._recorder._exit_subgraph()


        # add the loop operation to the graph
        #NOTE: this only exposes outputs of operations, not variables created within the loop
        self.op = Loop(
            external_inputs, 
            self.iter2_outputs, 
            self._graph, 
            self.vals, 
            self.iteration_variable, 
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
        return self.iteration_variable
        
    def __iter__(self):
        return self



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