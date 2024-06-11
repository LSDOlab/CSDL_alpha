import numpy as np
import pytest
x_shape = (3,2)
x_size = np.prod(x_shape)
def define_function():
    import csdl_alpha as csdl
    x = csdl.Variable(shape = x_shape, name = 'x', value = np.arange(x_size).reshape(x_shape))
    u = csdl.Variable(shape = x_shape, name = 'u', value = np.arange(x_size).reshape(x_shape)+1.0)
    y = x**2 + u
    w = (y/u)
    y.add_name('y')
    w.add_name('w')

    return (x, u), (y, w)

def test_finite_difference():
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()

    import csdl_alpha as csdl
    recorder = csdl.Recorder(inline = True)
    recorder.start()
    (x,u), (y, w) = define_function()

    from csdl_alpha.backends.jax.graph_to_jax import create_jax_interface

    jax_eval = create_jax_interface(
        inputs = [x,u],
        outputs = [y,w],
        graph = recorder.active_graph,
    )

    fd_inline = csdl.derivative_utils.finite_difference(
        ofs = (y,w),
        wrts = (x,u),
        forward_evaluation = None,
        step_size = 1e-6,
    )
    fd_jax = csdl.derivative_utils.finite_difference(
        ofs = (y,w),
        wrts = (x,u),
        forward_evaluation = jax_eval,
        step_size = 1e-3,
    )
    recorder.stop()

    analytical = {}
    analytical[y,x] = np.diag(2*x.value.flatten())
    analytical[y,u] = np.eye(x_size)
    analytical[w,x] = 2*np.diag((x.value/u.value).flatten())
    analytical[w,u] = np.diag(-(x.value.flatten()**2.0/u.value.flatten()**2.0))

    for fd in [fd_inline, fd_jax]:
        np.testing.assert_almost_equal(fd[y,x],analytical[y,x], decimal = 3)
        np.testing.assert_almost_equal(fd[y,u],analytical[y,u], decimal = 3)
        np.testing.assert_almost_equal(fd[w,x],analytical[w,x], decimal = 3)
        np.testing.assert_almost_equal(fd[w,u],analytical[w,u], decimal = 3)

def test_verify():
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()

    import csdl_alpha as csdl
    recorder = csdl.Recorder(inline = True)
    recorder.start()
    (x,u), (y, w) = define_function()

    deriv_values = {}
    deriv_values[y,x] = {}
    deriv_values[y,x]['value'] = 2*np.diag((x.value/u.value).flatten())
    deriv_values[y,x]['fd_value'] = 2*np.diag((x.value/u.value).flatten())+1e-7

    csdl.derivative_utils.verify_derivative_values(
        deriv_values
    )

    with pytest.raises(TypeError):
        deriv_values = 1
        csdl.derivative_utils.verify_derivative_values(
            deriv_values
        )

    with pytest.raises(TypeError):
        deriv_values = {}
        deriv_values[1] = {}
        csdl.derivative_utils.verify_derivative_values(
            deriv_values
        )

    with pytest.raises(TypeError):
        deriv_values = {}
        deriv_values[y,x] = 2.0
        csdl.derivative_utils.verify_derivative_values(
            deriv_values
        )
    with pytest.raises(KeyError):
        deriv_values = {}
        deriv_values[y,x] = {}
        deriv_values[y,x]['val'] = 0.0
        deriv_values[y,x]['fd_value'] = 0.0
        csdl.derivative_utils.verify_derivative_values(
            deriv_values
        )
    with pytest.raises(TypeError):
        deriv_values = {}
        deriv_values[y,x] = {}
        deriv_values[y,x]['value'] = np.array([0.0])
        deriv_values[y,x]['fd_value'] = 0.0
        csdl.derivative_utils.verify_derivative_values(
            deriv_values
        )

if __name__ == '__main__':
    test_finite_difference()
    test_verify()