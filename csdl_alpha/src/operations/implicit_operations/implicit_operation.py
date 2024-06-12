
from csdl_alpha.src.operations.operation_subclasses import SubgraphOperation
# TODO: make variable and operation easier to import

class ImplicitOperation(SubgraphOperation):

    def __init__(self, *args, name = 'nl_op', **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.name = name
        self.nonlinear_solver = self.metadata['nonlinear_solver']

    def compute_inline(self, *args):

        # print(f'COMPUTING NLSOLVER INLINE: {self.nonlinear_solver.name}')
        self.nonlinear_solver.solve_implicit_inline(*args)
        
        return [output.value for output in self.outputs]
    
    def compute_jax(self, *args):
        # want to give the nonlinear solver a function that computes the residuals

        from csdl_alpha.backends.jax.graph_to_jax import create_jax_function
        # outputs have the states, residuals aren't an output
        # need to make states an input to the jax function, and residuals an output
        # states = list(self.nonlinear_solver.state_to_residual_map.keys())
        # residuals = list(self.nonlinear_solver.state_to_residual_map.values())

        states = []
        residuals = []
        for state, residual  in self.nonlinear_solver.state_to_residual_map.items():
            states.append(state)
            residuals.append(residual)

        output_state_indices = [self.outputs.index(state) for state in states]
        sorted_output_state_indices = output_state_indices.copy()
        sorted_output_state_indices.sort()
        non_state_output_vars = [output for output in self.outputs if output not in states]
        jax_fn_inputs = self.inputs + states
        jax_fn_outputs = non_state_output_vars + residuals

        # jax function will return the outputs and residuals given inputs and states.
        # TODO: maybe have this made by implicit_operation and passed in (or from nonlinear_solver)
        jax_function = create_jax_function(self._subgraph, jax_fn_outputs, jax_fn_inputs)
        def jax_residual_function(states):
            """Computes residuals given states"""
            return jax_function(*args, *states)[len(non_state_output_vars):]

        states = self.nonlinear_solver.solve_implicit_jax(jax_residual_function, self.inputs, *args)

        outputs = jax_function(*args, *states)[:len(non_state_output_vars)]
        for ind in sorted_output_state_indices:
            i = output_state_indices.index(ind)
            outputs.insert(ind, states[i]) # (TODO: SLOW)
        return tuple(outputs)
    
    def prep_vjp(self):
        """
        Prepare operation for reverse mode differentiation
        """
        self.nonlinear_solver.prep_vjp()

    def evaluate_vjp(self, cotangents, *inputs_and_outputs):
        inputs = inputs_and_outputs[:len(self.inputs)]
        outputs = inputs_and_outputs[len(self.inputs):]

        # print('inputs           ', *(input.name for input in inputs))
        # print('inputs           ', *(input.name for input in self.inputs))
        # print('outputs          ', *(output.name for output in self.outputs))
        assert [*(input.name for input in inputs)] == [*(input.name for input in self.inputs)]
        assert [*(output.name for output in outputs)] == [*(output.name for output in self.outputs)]

        import csdl_alpha as csdl
        inputs_to_accumulate = []
        outputs_with_cotangents = []
        for input in inputs:
            if cotangents.check(input):
                inputs_to_accumulate.append(input)
        for output in outputs:
            if cotangents.check(output):
                outputs_with_cotangents.append(output)

        # print('needed inputs    ', *(input.name for input in inputs_to_accumulate))
        # print('needed outputs   ', *(output.name for output in outputs_with_cotangents))

        self.nonlinear_solver.accumulate_cotangents(cotangents, outputs_with_cotangents, inputs_to_accumulate)

        # raise NotImplementedError('Need to implement VJP for ImplicitOperation')