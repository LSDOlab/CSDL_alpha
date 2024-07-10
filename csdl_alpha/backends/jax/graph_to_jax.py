from ...src.graph.graph import Graph
from ...src.graph.variable import Variable
from ...utils.inputs import listify_variables

from csdl_alpha.src.operations.loops.loop import Loop
from csdl_alpha.src.operations.implicit_operations.implicit_operation import ImplicitOperation

import numpy as np
from typing import Union, Callable


# Get the graph
def get_jax_inputs(node, all_jax_variables:dict)->list:
    import jax.numpy as jnp
    jax_inputs = []
    for input in node.inputs:
        if input not in all_jax_variables:
            if input.value is None:
                raise ValueError(f"Jax function error with node {node}: Input {input} has no value")
            if input.value.dtype != np.float64:
                raise ValueError(f"Jax function error with node {node}: Expected input to be a float64, but got {input.value.dtype}")
            if isinstance(input.value, np.matrix):
                raise ValueError(f"Jax function error with node {node}: Expected input to be a float64, but got {input.value.dtype}")
            jax_inputs.append(jnp.array(input.value))
        else:
            jax_inputs.append(all_jax_variables[input])
    
    for i, input in enumerate(jax_inputs):
        if not isinstance(input, jnp.ndarray):
            raise ValueError(f"Jax function error with node {node}: Expected input to be a jnp.ndarray, but got {type(input)}")

    return jax_inputs

def update_jax_variables(node, jax_outputs, all_jax_variables:dict):
    import jax.numpy as jnp
    if not isinstance(jax_outputs, tuple):
        raise ValueError(f"Jax function error with node {node}: Expected output to be a tuple, but got {type(jax_outputs)}")
    for i, output in enumerate(jax_outputs):
        if not isinstance(output, jnp.ndarray):
            raise ValueError(f"Jax function error with node {node}: Expected output to be a jnp.ndarray, but got {type(output)}")
    if len(node.outputs) != len(jax_outputs):
        raise ValueError(f"Jax function error with node {node}: Expected {len(node.outputs)} outputs, but got {len(jax_outputs)}")

    for i, output in enumerate(node.outputs):
        try:
            all_jax_variables[output] = (jax_outputs[i]).reshape(output.shape)
        except:
            raise ValueError(f"Error updating JAX variables for node {node.name}. Output shape: {output.shape}, JAX output shape: {jax_outputs[i].shape}")

def create_jax_function(
        graph:Graph,
        outputs:list[Variable],
        inputs:list[Variable],
        )->callable:
    """Builds a JAX callable function from a CSDL graph.

    Parameters
    ----------
    rec : csdl.Recorder
    outputs : list[csdl.Variable]
        csdl variables to be returned by the function
    inputs : list[csdl.Variable]
        csdl variables to be passed to the function

    Returns
    -------
    callable
        A JAX function that takes in the inputs and returns the outputs
    """
    current_graph = graph
    import jax.numpy as jnp
    
    inputs = list(inputs)
    outputs = list(outputs)

    # Get the graph
    inputs = listify_variables(inputs)
    outputs = listify_variables(outputs)

    # Figure out the order to execute the graph
    import rustworkx as rx
    for output in outputs:
        if output not in current_graph.node_table:
            raise ValueError(f"Output {output} not in the graph")
    for input in inputs:
        if input not in current_graph.node_table:
            raise ValueError(f"Input {input} not in the graph")
    
    all_sorted_node_indices = rx.topological_sort(current_graph.rxgraph)
    all_sorted_nodes = [current_graph.rxgraph[i] for i in all_sorted_node_indices]
    sorted_nodes:list = [node for node in all_sorted_nodes]
    # print('FUNCTION:', len(inputs), len(outputs), len(sorted_nodes), current_graph.name)
    
    # Experiment?
    # from csdl_alpha.src.operations.derivatives.bookkeeping import build_derivative_node_order
    # all_sorted_nodes = build_derivative_node_order(current_graph, outputs, inputs, reverse=False)
    
    # Build the JAX function itself
    def jax_function(*args)->list:
        # Set the input values
        all_jax_variables = {}
        relevant_nodes = set()
        for node, arg in zip(inputs, args):
            all_jax_variables[node] = arg

        # Loop through each node in the graph in order and compute the JAX values
        for node in sorted_nodes:
            if isinstance(node, Variable):
                if node not in all_jax_variables:
                    relevant_nodes.add(node)
            
        for node in sorted_nodes:
            if isinstance(node, Variable):
                continue

            jax_inputs = get_jax_inputs(node, all_jax_variables)

            # TODO: Finish proper implicit operation/loop output handling
            # if isinstance(node, (ImplicitOperation, Loop)):
            #     fill_outputs = {output: None for output in node.outputs if output in relevant_nodes}
            #     # fill_outputs = {output: None for output in node.outputs if current_graph.out_degree(output) > 0}
            #     node.evaluate_jax(*jax_inputs, fill_outputs = fill_outputs)

            #     for output_node in fill_outputs:
            #         if fill_outputs[output_node] is None:
            #             raise ValueError(f"Jax function error with node {node}: Output {output_node} was not filled")
            #         all_jax_variables[output_node] = fill_outputs[output_node]
            # else:
            jax_outputs = node.compute_jax(*jax_inputs) # EVERY CSDL OPERATIONS NEEDS THIS FUNCTION
            if isinstance(jax_outputs, jnp.ndarray):
                jax_outputs = (jax_outputs,)
            update_jax_variables(node, jax_outputs, all_jax_variables)

        # Return the outputs
        for output in outputs:
            if output not in all_jax_variables:
                all_jax_variables[output] = jnp.array(output.value)
        return [all_jax_variables[output] for output in outputs]
    
    return jax_function

def create_jax_interface(
        inputs:Union[Variable, list[Variable], tuple[Variable]],
        outputs:Union[Variable, list[Variable], tuple[Variable]],
        graph:Graph = None,
        device:str='gpu',
        name = 'jax_interface')->Callable[[dict[Variable, np.ndarray]], dict[Variable, np.ndarray]]:
    """_summary_

    Parameters
    ----------
    inputs : Union[Variable, list[Variable], tuple[Variable]]
        Inputs to the jax function
    outputs : Union[Variable, list[Variable], tuple[Variable]]
        Outputs to the jax function
    graph : Graph, optional
        which graph to create the jax interface for, by default None

    Returns
    -------
    jax interface: Callable
        A function with type signature: jax_interface(dict[Variable, np.array])->dict[Variable, np.array], where the input and output variables must match the inputs and outputs respectively.
    """
    import jax
    import csdl_alpha as csdl

    # import os
    # os.environ['XLA_FLAGS'] = (
    #     '--xla_gpu_triton_gemm_any=True '
    #     '--xla_gpu_enable_latency_hiding_scheduler=true '
    # )


    # preprocessing:
    inputs = listify_variables(inputs)
    outputs = listify_variables(outputs)
    if graph is None:
        graph = csdl.get_current_recorder().active_graph

    # Create the JAX function
    # Insert JAX preprocessing here:
    # enabling x64 etc
    jax.config.update("jax_enable_x64", True)

    # Option in the future?
    if device == 'gpu':
        try:
            device = jax.devices('gpu')[0]
        except Exception as e:
            print(f'GPU not found: \'{e}\', falling back to CPU')
            device = jax.devices('cpu')[0]
    elif device == 'cpu':
        device = jax.devices('cpu')[0]
    else:
        raise ValueError(f"Invalid device {device}")

    jax_function = create_jax_function(graph, outputs, inputs)
    # jax_function = print_trace_time(jax_function, name = name) #TODO: uncomment for timing trace
    # jax_grad = jax.jit(jax.jacrev(jax_function, argnums=[i for i in range(len(inputs))])) # TODO: add option for jax derivatives?
    jax_function = jax.jit(jax_function, device=device)

    # Create the JAX interface
    def jax_interface(inputs_dict:dict[Variable, np.ndarray])->dict[Variable, np.ndarray]:
        jax_interface_inputs = []
        # print('INPUTS:')
        for input_var in inputs:
            jax_interface_inputs.append(jax.numpy.array(inputs_dict[input_var]))
        jax_outputs = jax_function(*jax_interface_inputs)

        # jax_grad_outputs = jax_grad(*jax_interface_inputs)

        outputs_dict = {}
        # print('OUTPUTS:')
        for i, output in enumerate(outputs):
            outputs_dict[output] = np.array(jax_outputs[i])
            # print(outputs_dict[output])
        return outputs_dict
    
    return jax_interface

def print_trace_time(jax_function, name):
    import time
    def wrapped(*args):
        print(f'Starting JAX trace for function \'{name}\'')
        start = time.time()
        outs = jax_function(*args)
        end = time.time()
        print(f'Finished JAX trace for function \'{name}\' in {end-start} seconds')
        return outs
    return wrapped