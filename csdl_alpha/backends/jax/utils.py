from csdl_alpha.src.graph.operation import Operation
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