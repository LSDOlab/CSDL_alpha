import csdl_alpha as csdl
from csdl_alpha.src.graph.graph import Graph
from csdl_alpha.src.recorder import Recorder
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import listify_variables

import numpy as np
from typing import Union


# Get the graph
def get_jax_inputs(node, all_jax_variables:dict)->list:
    import jax.numpy as jnp
    jax_inputs = []
    for input in node.inputs:
        if input not in all_jax_variables:
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
        all_jax_variables[output] = (jax_outputs[i]).reshape(output.shape)

def create_jax_function(
        graph:Graph,
        outputs:list[csdl.Variable],
        inputs:list[csdl.Variable],
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
    
    # Get the graph

    # Figure out the order to execute the graph
    import rustworkx as rx
    all_sorted_node_indices = rx.topological_sort(current_graph.rxgraph)
    all_sorted_nodes = [current_graph.rxgraph[i] for i in all_sorted_node_indices]
    sorted_nodes:list = [node for node in all_sorted_nodes if not isinstance(node, csdl.Variable)]
    
    # Build the JAX function itself
    def jax_function(*args)->list:
        # Set the input values
        all_jax_variables = {}
        for node, arg in zip(inputs, args):
            all_jax_variables[node] = arg

        # Loop through each node in the graph in order and compute the JAX values
        for node in sorted_nodes:
            jax_inputs = get_jax_inputs(node, all_jax_variables)
            jax_outputs = node.compute_jax(*jax_inputs) # EVERY CSDL OPERATIONS NEEDS THIS FUNCTION
            if not isinstance(jax_outputs, tuple):
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
        graph:Graph = None)->callable:
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
    jax interface: callable
        A function with type signature: jax_interface(dict[Variable, np.array])->dict[Variable, np.array], where the input and output variables must match the inputs and outputs respectively.
    """
    import jax

    # preprocessing:
    inputs = listify_variables(inputs)
    outputs = listify_variables(outputs)
    if graph is None:
        graph = csdl.get_current_recorder().active_graph

    # Create the JAX function
    # Insert JAX preprocessing here:
    # enabling x64 etc
    jax.config.update("jax_enable_x64", True)

    jax_function = create_jax_function(graph, outputs, inputs)
    jax_function = jax.jit(jax_function)

    # Create the JAX interface
    def jax_interface(inputs_dict:dict[Variable, np.array])->dict[Variable, np.array]:
        jax_interface_inputs = []
        # print('INPUTS:')
        for input_var in inputs:
            jax_interface_inputs.append(jax.numpy.array(inputs_dict[input_var]))
        jax_outputs = jax_function(*jax_interface_inputs)

        outputs_dict = {}
        # print('OUTPUTS:')
        for i, output in enumerate(outputs):
            outputs_dict[output] = np.array(jax_outputs[i])
            # print(outputs_dict[output])
        return outputs_dict
    
    return jax_interface
