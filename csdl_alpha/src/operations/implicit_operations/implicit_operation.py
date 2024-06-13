
from csdl_alpha.src.operations.operation_subclasses import SubgraphOperation
# TODO: make variable and operation easier to import

class ImplicitOperation(SubgraphOperation):

    def __init__(self, *args, name = 'nl_op', **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from .nonlinear_solvers.nonlinear_solver import NonlinearSolver
        self.name = name
        self.nonlinear_solver:NonlinearSolver = self.metadata['nonlinear_solver']

    def compute_inline(self, *args):

        # print(f'COMPUTING NLSOLVER INLINE: {self.nonlinear_solver.name}')
        self.nonlinear_solver.solve_implicit_inline(*args)
        
        return [output.value for output in self.outputs]
    
    def compute_jax(self, *args):
        """Computes the outputs of the operation using JAX

        Returns
        -------
        tuple
            Outputs of the operation
        """
        from csdl_alpha.backends.jax.graph_to_jax import create_jax_function, create_jax_interface

        # 1) Construct a JAX function that computes the residuals given the states
        # 2) Hand the JAX function and inputs to the nonlinear solver to solve for the states
        # 3) Compute the outputs using the solved states

        # outputs have the states, residuals aren't an output
        # need to make states an input to the jax function, and residuals an output

        state_vars = list(self.nonlinear_solver.state_to_residual_map.keys())
        residuals = list(self.nonlinear_solver.state_to_residual_map.values())
        non_state_output_vars = [output for output in self.outputs if output not in state_vars]

        jax_fn_inputs = self.inputs + state_vars
        jax_fn_outputs = non_state_output_vars + residuals

        # (1)
        jax_function = create_jax_function(self._subgraph, jax_fn_outputs, jax_fn_inputs)
        def jax_residual_function(states):
            """Computes residuals given states, ordered as in state_to_residual_map"""
            return jax_function(*args, *states)[len(non_state_output_vars):]

        # (2)
        input_dict = {input: arg for input, arg in zip(self.inputs, args)}
        states = self.nonlinear_solver.solve_implicit_jax(jax_residual_function, input_dict)
        state_dict = {state: states[i] for i, state in enumerate(state_vars)}

        # (3)
        outputs = jax_function(*args, *states)[:len(non_state_output_vars)]
        output_list = []
        ind = 0
        for output in self.outputs:
            try:
                output_list.append(state_dict[output])
            except:
                output_list.append(outputs[ind])
                ind += 1

        return tuple(output_list)
    
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