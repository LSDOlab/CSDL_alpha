import csdl_alpha as csdl
from csdl_alpha.src.graph.graph import Graph


# Get the graph
def get_jax_inputs(node, all_jax_variables:dict)->list:
    import jax.numpy as jnp
    jax_inputs = []
    for input in node.inputs:
        if input not in all_jax_variables:
            jax_inputs.append(jnp.array(input.value))
        else:
            jax_inputs.append(all_jax_variables[input])
    return jax_inputs

def update_jax_variables(node, jax_outputs, all_jax_variables:dict):
    for i, output in enumerate(node.outputs):
        all_jax_variables[output] = jax_outputs[i]

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
        return [all_jax_variables[output] for output in outputs]
    
    return jax_function
