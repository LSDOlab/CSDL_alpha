from csdl_alpha.src.operations.loops.loop import Loop, IterationVariable
from csdl_alpha.src.graph.variable import Variable
import numpy as np

def get_feedback_variables_and_stacks(loop_operation:Loop)->tuple[list[Variable], list[Variable]]:

    loop_vars = loop_operation.loop_vars
    loop_outputs = []
    loop_stacks = []
    for i in range(int(len(loop_vars)/2)):
        loop_outputs.append(loop_vars[i][2])
    for i in range(int(len(loop_vars)/2), len(loop_vars)):
        loop_stacks.append(loop_vars[i][2])
    return loop_outputs, loop_stacks

class ParentIOData():

    def __init__(
            self,
            external_body_IO:Variable,
            ):
        # Store primal variable
        self.external_body_IO:Variable = external_body_IO
        
        # Store cotangent feedbacks
        self.out_cotangent:Variable = None
        self.external_input_cotangent:Variable = None
        self.body_input_cotangent:Variable = None

def build_external_inputs_data(
        loop_operation:Loop,
        feedback_inputs:set[Variable],
        cotangents,)->list[ParentIOData]:
    parent_external_inputs:list[Variable] = []
    for input in loop_operation.inputs:
        # TODO: Double check correctness of this condition
        # if input in loop_operation.get_subgraph().node_table:

        if input not in feedback_inputs:
            if cotangents.check(input):
                input_data = ParentIOData(input)
                input_data.external_input_cotangent = Variable(
                    name = f'{input.name}_ext_cot_in',
                    value = np.zeros(input.shape),
                )
                parent_external_inputs.append(input_data)
    return parent_external_inputs

def build_external_outputs_data(
        loop_operation:Loop,
        feedback_outputs:set[Variable],
        cotangents,)->list[ParentIOData]:
    parent_external_outputs:list[Variable] = []
    for output in loop_operation.outputs:
        if output not in feedback_outputs:
            if cotangents.check(output):
                output_data = ParentIOData(output)
                output_data.external_input_cotangent = cotangents[output]
                parent_external_outputs.append(output_data)
    return parent_external_outputs

class FeedBackData():

    def __init__(
            self,
            external_input:Variable,
            body_input:Variable,
            input_stack:Variable,
            body_external_output:Variable,
            ):
        self.external_input:Variable = external_input
        self.body_input:Variable = body_input
        self.input_stack:Variable = input_stack
        self.body_external_output:Variable = body_external_output
        
        self.out_cotangent:Variable = None
        self.external_input_cotangent:Variable = None
        self.body_input_cotangent:Variable = None

def build_feedback_data(loop_operation:Loop, cotangents)->list[FeedBackData]:
    feedbacks = []

    loop_vars = loop_operation.loop_vars

    num_feedbacks = int(len(loop_vars))
    for i in range(num_feedbacks):
        fbd = FeedBackData(
            external_input = loop_vars[i][1],
            body_input = loop_vars[i][0],
            input_stack = loop_operation.outputs[-(num_feedbacks-i)],
            body_external_output = loop_vars[i][2]
        )

        if cotangents.check(fbd.body_external_output):
            fbd.external_input_cotangent = cotangents[fbd.body_external_output]
        else:
            fbd.external_input_cotangent = Variable(value = np.zeros(fbd.external_input.shape))
        feedbacks.append(fbd)
    return feedbacks

def build_reversed_iteration_variables(
        parent_iter_vars:list[IterationVariable]
        )->tuple[IterationVariable,list[IterationVariable]]:
    
    # Build the reversed iteration variables
    reversed_iter_vars:list[IterationVariable] = []
    for orig_iter_var in parent_iter_vars:
        reversed_vals = list(reversed(orig_iter_var.vals))
        reversed_iter_vars.append(IterationVariable(vals = reversed_vals))
    
    # Build reversed indexing variable
    reversed_range = list(reversed(range(len(parent_iter_vars[0].vals))))
    return IterationVariable(vals = reversed_range), reversed_iter_vars