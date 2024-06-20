from csdl_alpha.src.graph.variable import Variable
import numpy as np

class StaticInputData():
    def __init__(
            self,
            variable:Variable,
            ):
        self.external_body_input:Variable = variable
        self.external_cotangent:Variable = None
        self.internal_cotangent:Variable = None

class BatchedData():
    def __init__(
            self,
            batched:Variable,
            unbatched:Variable,
        ):
        self.batched:Variable = batched
        self.unbatched:Variable = unbatched
        
        self.batched_cotangent:Variable = None
        self.unbatched_cotangent:Variable = None

def build_static_input_data(loop_operation, cotangents)->tuple[list[StaticInputData],list[Variable]]:
    parent_static_inputs = []
    remaining_parent_static_inputs = []
    for input in loop_operation.unbatched_inputs:
        if cotangents.check(input):
            input_data = StaticInputData(input)
            input_data.external_cotangent = Variable(
                    name = f'{input.name}_ext_cot',
                    value = np.zeros((loop_operation.batch_size, *input.shape)),
                )
            parent_static_inputs.append(input_data)
        else:
            remaining_parent_static_inputs.append(input)
    return parent_static_inputs, remaining_parent_static_inputs

def build_batched_input_data(loop_operation, cotangents)->tuple[list[BatchedData],list[tuple[Variable,Variable]]]:
    parent_batched_inputs = []
    remaining_parent_batched_inputs = []
    for batched, unbatched in zip(loop_operation.outer_batched_inputs, loop_operation.inner_unbatched_inputs):
        if cotangents.check(batched):
            batched_data = BatchedData(batched, unbatched)
            batched_data.batched_cotangent = Variable(
                    name = f'{batched.name}_ext_cot',
                    value = np.zeros(batched.shape),
                )
            parent_batched_inputs.append(batched_data)
        else:
            remaining_parent_batched_inputs.append((batched, unbatched))
    return parent_batched_inputs, remaining_parent_batched_inputs

def build_batched_output_data(loop_operation, cotangents)->tuple[list[BatchedData],list[tuple[Variable,Variable]]]:
    parent_batched_outputs = []
    remaining_parent_batched_outputs = []
    for batched, unbatched in zip(loop_operation.outer_batched_outputs, loop_operation.inner_unbatched_outputs):
        if cotangents.check(batched):
            batched_data = BatchedData(batched, unbatched)
            batched_data.batched_cotangent = cotangents[batched]
            parent_batched_outputs.append(batched_data)
        else:
            remaining_parent_batched_outputs.append((batched, unbatched))
    return parent_batched_outputs, remaining_parent_batched_outputs