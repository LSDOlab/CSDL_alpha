from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.recorder import Recorder

def fallback_to_inline_jax(operation:Operation, *args:list['jnp.array'])->tuple['jnp.array']:
    '''
    If the operation has no jax implementation, fall back to the inline implementation.
    '''
    import jax
    import numpy as np

    def new_inline_func(*args_in):
        processed_inputs = [np.array(input) for input in args_in]
        return operation.compute_inline(*processed_inputs)

    output = jax.pure_callback(
        new_inline_func,
        [jax.ShapeDtypeStruct(output.shape, np.float64) for output in operation.outputs],
        *args)
    return tuple(output)


def verify_inline_vs_jax(
        recorder:Recorder,
        abs_threshold:float=None,
        rel_threshold:float=None,
        stride = 1,
        raise_error:bool=True,
        print_errors:bool=True,
    ):
    insights = recorder.gather_insights()
    root_graph = recorder.get_root_graph()

    all_inputs = [input for input in insights['input_nodes']]

    node_num = 0
    all_outputs = []
    for node in root_graph.node_table:
        if not isinstance(node, Operation):
            if node not in insights['input_nodes']:
                if node_num % stride == 0:
                    all_outputs.append(node)
                node_num += 1
    
    from csdl_alpha.backends.jax.graph_to_jax import create_jax_interface

    jax_func = create_jax_interface(
        inputs=all_inputs,
        outputs=all_outputs,
    )

    input_values = {input: input.value for input in all_inputs}
    output_values = jax_func(input_values)

    recorder.execute()

    import numpy as np
    import rustworkx as rx
    all_sorted_node_indices = rx.topological_sort(root_graph.rxgraph)
    all_sorted_nodes = [root_graph.rxgraph[i] for i in all_sorted_node_indices]

    return_error = {}
    for node in all_sorted_nodes:
        if node not in all_outputs:
            continue

        if node.value is None:
            continue

        return_error[node] = {
            'inline': node.value,
            'jax': output_values[node],
            'error': node.value - output_values[node],
        }
        error = np.linalg.norm(node.value - output_values[node])
        rel_error = error/np.linalg.norm(node.value)
        past_threshhold = False
        if abs_threshold is not None:
            if error > abs_threshold:
                past_threshhold = True
        if rel_threshold is not None:
            if rel_error > rel_threshold and np.linalg.norm(node.value) > 1e-7:
                past_threshhold = True
        if past_threshhold:
            if raise_error:
                print_errors = True
            if print_errors:
                print(f'\nHigh discrepancy with variable: {node.name} of shape ({node.shape}) ({node})')
                print('absolute error:        ', error)
                print('relative error:        ', rel_error)
                print('max absolute error:    ', np.max(np.abs(error)))
                print('avg jax value:         ', np.mean(output_values[node]))
                print('avg inline value:      ', np.mean(node.value))
                print('max jax value:         ', np.max(output_values[node]))
                print('max inline value:      ', np.max(node.value))
                print('min jax value:         ', np.min(output_values[node]))
                print('min inline value:      ', np.min(node.value))
                print('trace:')
                node.print_trace()
                print(f'predecessors: {root_graph.predecessors(node)}')
                print(f'{node.value}')
                print(f'{output_values[node]}')
                print('==============================ERROR==============================')

            if raise_error:
                raise ValueError(f'Error in variable {node.name} of shape {node.shape} with abs error {error} and rel error {rel_error}. (More information above ^^^)')

        else:
            if print_errors:
                print(node.name, np.linalg.norm(node.value - output_values[node]))
    return return_error