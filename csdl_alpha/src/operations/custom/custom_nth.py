from csdl_alpha.utils.parameters import Parameters
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.utils.inputs import variablize, get_type_string, ingest_value
from csdl_alpha.src.operations.custom.utils import (
    prepare_compute_derivatives,
    process_custom_derivatives_metadata,
    postprocess_compute_derivatives,
    preprocess_custom_inputs,
    postprocess_custom_outputs,
)
from csdl_alpha.src.operations.custom.custom import CustomExplicitOperation, CustomOperation

import warnings
import numpy as np
from typing import Union, Dict, Tuple, Optional, Callable
import inspect

class CustomExplicitOperationBeta(CustomExplicitOperation):

    def __init__(self):
        super().__init__()

    def evaluate(self):
        raise NotImplementedError('not implemented')

    def compute(self, inputs, outputs):
        raise NotImplementedError('not implemented')

    def compute_derivatives(self, inputs, outputs, derivatives, order):
        raise NotImplementedError(f'not implemented for operation {self.name}')
    
    def declare_derivative_parameters(self, output_name, input_name, **kwargs):
       raise NotImplementedError('Use self.declare_derivative_function instead or a CustomExplicitOperation')
    
    def declare_derivative_function(self, derivative_operation:Union[Callable, type[CustomOperation]], *args, **kwargs):
        if inspect.isclass(derivative_operation) and issubclass(derivative_operation, CustomOperation):
            call_deriv_func = lambda inputs: derivative_operation(*args, **kwargs).evaluate(inputs)
        elif callable(derivative_operation):
            call_deriv_func = lambda inputs: derivative_operation(inputs, *args, **kwargs)
        else:
            raise TypeError(f'derivative_operation must be a callable function or a CustomOperation class (not instance), got type {get_type_string(derivative_operation)}')
        
        def vjp_func(cotangents:dict, inputs:dict[Variable]):
            # call derivative function
            jacobians = call_deriv_func(inputs)

            # postprocess output functions and check jacobians dictionary to make sure everything is correct
            jacobians, info = postprocess_custom_nth_derivs(jacobians)

            # accumulate cotangents
            print('poo')

        self.vjp_func = vjp_func
        
    def evaluate_vjp(self, cotangents, *inputs_and_outputs):
        inputs = inputs_and_outputs[:self.num_inputs]

        # dictionify inputs to derivative function
        input_dict = {}
        for i, (input_name, original_input) in enumerate(self.input_dict.items()):
            input_dict[input_name] = inputs[i]

        # call derivative function
        self.vjp_func(cotangents, input_dict)


        # output_cots = []
        # for output in outputs:
        #     if not cotangents.check(output):
        #         output_cots.append(Variable(value = np.zeros(output.shape)))
        #     else:
        #         output_cots.append(cotangents[output])
        # input_cots = []
        # for input in inputs:
        #     if cotangents.check(input):
        #         input_cots.append(input)

        # vjps = self.build_custom_operation_vjp(
        #     input_cotangents = input_cots,
        #     output_cotangents = output_cots,
        #     deriv_order = 1)
        # cots = vjps.finalize_and_return_outputs()
        
        # if not isinstance(cots, tuple):
        #     cots = (cots,)
        # for i, input in enumerate(input_cots):
            # cotangents.accumulate(input, cots[i])
