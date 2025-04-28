from csdl_alpha.utils.parameters import Parameters
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.utils.inputs import variablize, get_type_string, ingest_value
from csdl_alpha.src.operations.custom.utils import (
    postprocess_custom_nth_derivs
)
from csdl_alpha.src.operations.custom.custom import CustomExplicitOperation, CustomOperation
from csdl_alpha.src.operations.derivatives.bookkeeping import VarTangents

import warnings
import numpy as np
from typing import Union, Dict, Tuple, Optional, Callable
import inspect

class CustomExplicitOperationBeta(CustomExplicitOperation):

    def __init__(self):
        super().__init__()
        self.vjp_func = None

    def evaluate(self):
        raise NotImplementedError('not implemented')

    def compute(self, inputs, outputs):
        raise NotImplementedError('not implemented')

    def compute_derivatives(self, inputs, outputs, derivatives, order):
        raise NotImplementedError(f'not implemented for operation {self.name}')
    
    def declare_derivative_parameters(self, output_name, input_name, **kwargs):
       raise NotImplementedError('Use self.declare_derivative_function instead or a CustomExplicitOperation')
    
    def declare_derivative_function(self, derivative_operation:Union[Callable, type[CustomOperation]], *args, **kwargs):
        """Declare a custom operation (or function) that computes the derivatives of this custom operation.

        Parameters
        ----------
        derivative_operation : Union[Callable, type[CustomOperation]]
        """
        
        if inspect.isclass(derivative_operation) and issubclass(derivative_operation, CustomOperation):
            call_deriv_func = lambda inputs: derivative_operation(*args, **kwargs).evaluate(inputs)
        elif callable(derivative_operation):
            call_deriv_func = lambda inputs: derivative_operation(inputs, *args, **kwargs)
        else:
            raise TypeError(f'derivative_operation must be a callable function or a CustomOperation class (not instance), got type {get_type_string(derivative_operation)}')
        
        def vjp_func(cotangents:VarTangents, inputs:list[Variable], outputs:list[Variable]):
            # dictionify inputs to derivative function
            input_dict = {}
            for i, (input_name, original_input) in enumerate(self.input_dict.items()):
                input_dict[input_name] = inputs[i]
            
            # call derivative function
            jacobians = call_deriv_func(input_dict)

            # postprocess output functions and check jacobians dictionary to make sure everything is correct
            jacobians = postprocess_custom_nth_derivs(
                jacobians,
                self.input_dict,
                self.output_dict,
            )

            # accumulate cotangents
            for i, (input_name, input) in enumerate(self.input_dict.items()):
                inputs_vjp = inputs[i]
                if not cotangents.check(inputs_vjp):
                    continue
                for j, (output_name, output) in enumerate(self.output_dict.items()):
                    output_vjp = outputs[j]
                    if not cotangents.check(output_vjp):
                        continue
                    jac = jacobians[output_name, input_name]
                    if jac is None:
                        continue

                    cotangents.accumulate(inputs_vjp, (cotangents[output_vjp].reshape(1,-1)@jac).reshape(inputs_vjp.shape))

        self.vjp_func = vjp_func
        
    def evaluate_vjp(self, cotangents:VarTangents, *inputs_and_outputs):
        inputs = inputs_and_outputs[:self.num_inputs]
        outputs = inputs_and_outputs[self.num_inputs:]

        # call derivative function
        if self.vjp_func is None:
            raise RuntimeError(f'Derivatives for custom operation {self.info()} has not been set. Use self.declare_derivative_function to define derivatives.')
        self.vjp_func(cotangents, inputs, outputs)
