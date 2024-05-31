from csdl_alpha.src.operations.custom.custom import CustomExplicitOperation, CustomJacOperation
import numpy as np
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, get_type_string, ingest_value
from csdl_alpha.src.operations.custom.utils import (
    preprocess_custom_inputs,
    postprocess_custom_outputs,
)
class CustomImplicitOperation(CustomExplicitOperation):
    
    def compute_forward(self, inputs, outputs):
        self.solve_residual_equations(inputs, outputs)

    def build_custom_operation_vjp(
            self,
            input_cotangents:list[Variable],
            output_cotangents:list[Variable],
            deriv_order:int)->'CustomJacOperation':

        if deriv_order > 1:
            raise NotImplementedError('Higher order custom derivatives not yet implemented')

        return CustomImplicitJacOperation(self, input_cotangents, output_cotangents, deriv_order)

    def solve_residual_equations(self, inputs, outputs):
        raise NotImplementedError('compute_residual_equations method must be implemented in a custom implicit operation')
    
    def apply_inverse_jacobian(self, inputs, outputs, d_outputs, d_residuals, mode):
        # for mode = rev:
        # d_outputs --> d_residuals
        raise NotImplementedError('apply_inverse_jacobian method must be implemented in a custom implicit operation')

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        # for mode = rev
        # d_residuals --> d_inputs
        raise NotImplementedError('compute_jacvec_product method must be implemented in a custom implicit operation')

class CustomImplicitJacOperation(CustomJacOperation):
    
    def compute_inline(self, *orig_inputs_and_outputs_and_cots:list[np.array])->list[np.array]:
        """Perform the derivative accumulation procedure here.
        Two main steps:
        1. Call the user's compute_derivatives method and retrieve jacobians
        -- If the original input values are the same as the previous execution, we can use the previous jacobians
        2. Accumulate the cotangents using simple matrix vector products
        """
        input_values:list[np.array] = orig_inputs_and_outputs_and_cots[:self.num_orig_inputs]
        output_values:list[np.array] = orig_inputs_and_outputs_and_cots[self.num_orig_inputs:self.num_orig_inputs + self.num_orig_outputs]
        cot_values:list[np.array] = orig_inputs_and_outputs_and_cots[self.num_orig_inputs + self.num_orig_outputs:]
        
        inputs:dict[str,Variable] = {self.reverse_input_dict[key]:input for key, input in zip(self.orig_inputs, input_values)}
        outputs:dict[str,Variable] = {self.reverse_output_dict[key]:output for key, output in zip(self.orig_outputs, output_values)}
        d_outputs:dict[str,Variable] = {self.reverse_output_dict[key]:d_output for key, d_output in zip(self.orig_outputs, cot_values)}

        # Solve adjoint system
        d_residuals = {output_str:np.zeros(output.shape) for output_str, output in self.custom_operation.output_dict.items()}
        inputs = preprocess_custom_inputs(inputs)
        outputs = preprocess_custom_inputs(outputs)
        d_outputs = preprocess_custom_inputs(d_outputs)
        self.custom_operation.apply_inverse_jacobian(inputs, outputs, d_outputs, d_residuals, mode = 'rev')
        d_residuals = postprocess_custom_outputs(d_residuals, self.custom_operation.output_dict)
        
        # Compute the jacobian vector product
        d_inputs = {input_str:np.zeros(input.shape) for input_str, input in self.custom_operation.input_dict.items()}
        d_residuals = preprocess_custom_inputs(d_residuals)
        self.custom_operation.compute_jacvec_product(inputs, outputs, d_inputs, d_outputs, d_residuals, mode = 'rev')
        d_inputs = postprocess_custom_outputs(d_inputs, self.custom_operation.input_dict)

        # Accumulate and return
        input_cots:list[np.array] = []
        for input in self.input_cotangents:
            input_cots.append(-d_inputs[self.reverse_input_dict[input]].reshape(input.shape))

        if len(input_cots) == 1:
            return input_cots[0]
        else:
            return tuple(input_cots)