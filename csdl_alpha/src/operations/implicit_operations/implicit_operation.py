
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
    
    def evaluate_vjp(self, cotangents, *inputs_and_outputs):
        inputs = inputs_and_outputs[:len(self.inputs)]
        outputs = inputs_and_outputs[len(self.inputs):]

        import csdl_alpha as csdl

        print('inputs           ', *(input.name for input in self.inputs))
        print('outputs          ', *(output.name for output in self.outputs))

        inputs_to_accumulate = []
        for input in self.inputs:
            if cotangents.check(input):
                inputs_to_accumulate.append(input)
        
        print('needed inputs    ', *(input.name for input in inputs_to_accumulate))

        raise NotImplementedError('Need to implement VJP for ImplicitOperation')