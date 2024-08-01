from csdl_alpha.src.operations.loops.loop import IterationVariable
from csdl_alpha.src.operations.loops.new_loop.feedbacks import Feedback
from csdl_alpha.src.graph.variable import Variable

import numpy as np
from typing import Union

def build_iteration_variables(vals:Union[list[list[int]], list[int]])->dict['IterationVariable', list[int]]:
    if not isinstance(vals, (list, int, np.integer)):
        raise TypeError("vals must be a list of lists of integers or a list of integers or an integer or an integer")
    
    if isinstance(vals, (int, np.integer)):
        vals = [list(range(vals))]
    elif len(vals) == 0:
        raise ValueError("vals must not be empty")

    if not isinstance(vals[0], list):
        vals = [vals]

    iter_dict_var_dict = {}

    num_iterations = None
    for i, current_iter_vals in enumerate(vals):
        if not isinstance(current_iter_vals, list):
            raise TypeError(f"expected list but got {type(current_iter_vals)} for iter val {i}")

        if num_iterations is None:
            num_iterations = len(current_iter_vals)

        if len(current_iter_vals) != num_iterations:
            raise ValueError("all iteration variables must have the same number of values")

        iter_dict_var_dict[IterationVariable(vals = current_iter_vals)] = current_iter_vals

    return iter_dict_var_dict

def reverse_iteration_values(iter_vars:dict['IterationVariable', list[int]])->list[list[int]]:
    return [list(reversed(vals)) for iter_var, vals in iter_vars.items()]

class FeedbackDeriv():
    def __init__(self, feedback:Feedback):
        self.out_cotangent:Variable = None
        self.external_input_cotangent:Variable = None
        self.body_input_cotangent:Variable = None
        self.stacked_internal_input:Variable = None

        # parent feedback data
        self.parent_fb:Feedback = feedback

class ParentIODeriv():
    def __init__(
            self,
            external_body_IO:Variable,
            ):
        # Store forward variable
        self.external_body_IO:Variable = external_body_IO
        
        # Store cotangent feedbacks
        self.out_cotangent:Variable = None
        self.external_input_cotangent:Variable = None
        self.body_input_cotangent:Variable = None

class MetaOutputDeriv():
    def __init__(
            self,
            internal_target:Variable,
            external_output:Variable,
            ):
        # Store forward variabless
        self.internal_target:Variable = internal_target
        self.external_output:Variable = external_output
        
        # Store cotangent variables
        self.internal_target_cotangent:Variable = None
        self.external_output_cotangent:Variable = None

def build_loop_deriv_data(loop, cotangents):
    from csdl_alpha.src.operations.loops.new_loop.new_loop import NewLoop
    loop:NewLoop = loop

    # Process feedbacks
    # TODO: What do we do about the feedbacks external input if not in a normal input????
    feedback_data = {}
    for int_input, feedback in loop.loop_builder.feedbacks._int_input_to_feedback.items():
        feedback:Feedback = feedback
        feedback_data[int_input] = FeedbackDeriv(feedback)
        feedback_data[int_input].stacked_internal_input = loop.loop_builder.stacked[feedback.internal_input]
        feedback_data[int_input].external_input_cotangent = Variable(
            name = f'zero_cot_{feedback.output.name}',
            shape = feedback.output.shape,
            value = np.zeros(feedback.output.shape)
        )


    # Process all iteration variables
    iter_vars = loop.loop_builder.iters

    # Process inputs:
    input_data = {}
    for input in loop.loop_builder.inputs:
        if cotangents.check(input):
            input_data[input] = ParentIODeriv(input)
            input_data[input].external_input_cotangent = Variable(
                    name = f'{input.name}_ext_cot_in',
                    value = np.zeros(input.shape),
                )

    # Process outputs:
    std_output_data = {}
    for output in loop.loop_builder.outputs:
        if cotangents.check(output):
            std_output_data[output] = ParentIODeriv(output)
            std_output_data[output].external_input_cotangent = cotangents[output]

    # Process accrued outputs:
    accrued_output_data = {}
    for accrue_target, accrued_output in loop.loop_builder.accrued.items():
        if cotangents.check(accrued_output):
            accrued_output_data[accrue_target] = MetaOutputDeriv(accrue_target, accrued_output)
            accrued_output_data[accrue_target].external_output_cotangent = cotangents[accrued_output]

    # Process stacked outputs:
    stacked_output_data = {}
    for stack_target, stacked_output in loop.loop_builder.stacked.items():
        if cotangents.check(stacked_output):
            stacked_output_data[stack_target] = MetaOutputDeriv(stack_target, stacked_output)
            stacked_output_data[stack_target].external_output_cotangent = cotangents[stacked_output]

    return feedback_data, iter_vars, input_data, std_output_data, accrued_output_data, stacked_output_data

def add_to_seed_dict(seed_dict:dict[Variable,Variable], key:Variable, value:Variable):
    if key in seed_dict:
        seed_dict[key] = seed_dict[key] + value
    else:
        seed_dict[key] = value