
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