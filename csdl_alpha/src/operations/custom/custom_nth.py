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

import warnings
import numpy as np

class CustomExplicitOperationNew(CustomOperation):

    def __init__(self):
        super().__init__()

    def evaluate(self):
        raise NotImplementedError('not implemented')

    def compute(self, inputs, outputs):
        raise NotImplementedError('not implemented')

    def compute_derivatives(self, inputs, outputs, derivatives, order):
        raise NotImplementedError(f'not implemented for operation {self.name}')
    