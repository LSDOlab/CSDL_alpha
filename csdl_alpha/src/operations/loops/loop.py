from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.graph.variable import Variable

class IterationVariable(Variable):
    def __init__(self, vals):
        super().__init__(vals[0])
        self.shape = None
        self.vals = vals
        self.name = 'iter'

class LoopVariable(Variable):
    def __init__(self, var1, var2):
        super().__init__(shape=(var1.shape)) #NOTE: idk if we even need to do this
        self.shape=None
        self.var1 = var1
        self.var2 = var2

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
        self.loop_vars = loop_vars
        self._add_outputs_to_graph()
        self._add_to_graph()

    def _add_outputs_to_graph(self):
        for output in self.outputs:
            self.recorder.active_graph.add_node(output)

    def compute_inline(self):
        for i in range(len(self.vals)):
            self.iter_var.set_value(self.vals[i])
            for loop_var in self.loop_vars:
                if i == 0:
                    loop_var.set_value(loop_var.var1.value)
                else:
                    loop_var.set_value(loop_var.var2.value)
            self.graph.execute_inline()


class vrange():
    def __init__(self, lower=0, upper = 100, increment=1, *, vals = None):
        
        # Process runtime iterations
        if upper < lower:
            raise ValueError(f'The lower bound of the for loop, {lower}, is above the upper bound of the for loop, {upper}')
        self.curr_index = 0
        if type(vals)==type(None):
            self.vals = list(range(lower, upper, increment))
        else:
            self.vals = vals

        # enter new graph
        from csdl_alpha.api import manager
        self._recorder = manager.active_recorder
        self._recorder._enter_subgraph(add_missing_variables=True)
        self._graph = self._recorder.active_graph

        # initialize iteration variable:
        self.iteration_variable = IterationVariable(self.vals)


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

        # deleting the operations so we cana find inputs to the second iteration in the same way
        self._graph._delete_nodes(ops)

    def post_iteration_two(self):
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
        for input1, input2 in zip(self.iter1_inputs, self.iter2_inputs):
            if not input1 is input2: 
                if input2 in self.iter1_outputs:
                    # this implies input 1 comes from outside the loop and input 2 comes from the first iteration
                    # we want to go from input2 to the corresponding output of the 2nd iteration
                    output2 = self.iter2_outputs[self.iter1_outputs.index(input2)]
                    loop_var = LoopVariable(input1, output2)
                    loop_vars.append(loop_var)
                    self._graph._delete_nodes([input1])
                    self.iter1_non_inputs.remove(input2)
                    self._graph._replace_node(input2, loop_var)
                else:
                    # this implies input 1 and input 2 are both made in the loop, so we can just keep input 2
                    pass

        # delete any remnanats of the first iteration
        self._graph._delete_nodes(self.iter1_non_inputs)

        external_inputs = self._graph.inputs

        # Stop the graph
        self._graph.visualize('graph_loop_final')
        self._recorder._exit_subgraph()

        # add the loop operation to the graph
        #NOTE: this only exposes outputs of operations, not variables created within the loop
        op = Loop(external_inputs, self.iter2_outputs, self._graph, self.vals, self.iteration_variable, loop_vars)
        if self._recorder.inline:
            op.compute_inline()

    def __next__(self):
        if self.curr_index==1:
            self.post_iteration_one()
        elif self.curr_index == 2:
            self.post_iteration_two()
            raise StopIteration
        self.curr_index+=1
        return self.iteration_variable
        
    def __iter__(self):
        return self


    

if __name__ == '__main__':
    import csdl_alpha as csdl
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    a = csdl.Variable(value=2, name='a')
    b = csdl.Variable(value=3, name='b')
    for i in vrange(0, 10):
        b = a + b
        c = a*2

    print(b.value) # should be 23
    print(c.value) # should be 4
    recorder.active_graph.visualize('outer_graph')
    recorder.stop()