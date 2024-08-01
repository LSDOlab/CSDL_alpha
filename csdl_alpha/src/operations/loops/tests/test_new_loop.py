import pytest

def test_build_vals():
    from csdl_alpha.src.operations.loops.new_loop.utils import build_iteration_variables
    
    import csdl_alpha as csdl
    rec = csdl.Recorder()
    rec.start()
    iters = build_iteration_variables([[1, 2, 3], [4, 5, 6]])
    build_iteration_variables(1)
    with pytest.raises(ValueError):
        build_iteration_variables([])
    with pytest.raises(TypeError):
        build_iteration_variables([[1], 2])
    with pytest.raises(ValueError):
        build_iteration_variables([[1], [2, 3]])


def test_loop_indices():
    from csdl_alpha.src.operations.loops.new_loop.loop_builder import LoopBuilder
    from csdl_alpha.src.operations.loops.new_loop.utils import build_iteration_variables
    import csdl_alpha as csdl
    rec = csdl.Recorder()
    rec.start()
    iters = build_iteration_variables([[1, 2, 3], [4, 5, 6]])
    loop_builder = LoopBuilder(rec.active_graph, iters)
    i,j = loop_builder.get_loop_indices()
    assert i.vals == [1, 2, 3]
    assert j.vals == [4, 5, 6]

    rec = csdl.Recorder()
    rec.start()
    iters = build_iteration_variables([[1, 2, 3]])
    loop_builder = LoopBuilder(rec.active_graph, iters)
    i = loop_builder.get_loop_indices()
    assert i.vals == [1, 2, 3]

import csdl_alpha as csdl
import numpy as np

def compute_real(x0_val, a_val):
    last_x_np = np.zeros((1,))
    accrued_x_np = np.zeros((1,))
    stacked_x_np = np.zeros((5,1))
    stacked_x_new_np = np.zeros((5,1))
    x_np = x0_val
    for i in range(5):
        accrued_x_np += x_np
        stacked_x_np[i] = x_np
        x_np = x_np*a_val[i] + 1
        stacked_x_new_np[i] = x_np
        last_x_np = x_np

    return last_x_np, accrued_x_np, stacked_x_np, stacked_x_new_np

def test_newloop():
    rec = csdl.Recorder(inline = True, debug=True)
    rec.start()

    a_val = np.array([0.1, 0.2, 0.3, 0.4, -0.1])
    x0_val = np.array([2.0])
    x0 = csdl.Variable(name = 'x0', value = x0_val)
    a = csdl.Variable(name = 'a', value = a_val)

    with csdl.experimental.enter_loop(5) as loop_builder:
        i = loop_builder.get_loop_indices()
        x = loop_builder.initialize_feedback(x0)
        x_new = x*a[i] + 1
        loop_builder.finalize_feedback(x, x_new)

    accrued_x = loop_builder.add_pure_accrue(x)
    stacked_x = loop_builder.add_stack(x)
    stacked_x_new = loop_builder.add_stack(x_new)
    last_x = loop_builder.add_output(x_new)
    loop_builder.finalize(add_all_outputs = False)

    last_x_np, accrued_x_np, stacked_x_np, stacked_x_new_np = compute_real(x0_val, a_val)
    assert np.allclose(last_x_np, last_x.value)
    assert np.allclose(accrued_x_np, accrued_x.value)
    assert np.allclose(stacked_x_np, stacked_x.value)
    assert np.allclose(stacked_x_new_np, stacked_x_new.value)

    x0_val = np.array([3.0])
    a_val = np.array([0.2, 0.3, 0.6, 0.5, -0.2])
    x0.value = x0_val
    a.value = a_val
    rec.execute()
    last_x_np, accrued_x_np, stacked_x_np, stacked_x_new_np = compute_real(x0_val, a_val)
    assert np.allclose(last_x_np, last_x.value)
    assert np.allclose(accrued_x_np, accrued_x.value)
    assert np.allclose(stacked_x_np, stacked_x.value)
    assert np.allclose(stacked_x_new_np, stacked_x_new.value)

    jax_interface = csdl.jax.create_jax_interface(inputs = [x0, a], outputs = [last_x, accrued_x, stacked_x, stacked_x_new])
    outputs = jax_interface({x0: x0_val, a: a_val})
    assert np.allclose(last_x_np, outputs[last_x])
    assert np.allclose(accrued_x_np, outputs[accrued_x])
    assert np.allclose(stacked_x_np, outputs[stacked_x])
    assert np.allclose(stacked_x_new_np, outputs[stacked_x_new])
    
    outs = csdl.derivative_utils.verify_derivatives(
        ofs = [last_x, accrued_x, stacked_x, stacked_x_new],
        wrts = [x0, a],
        step_size=1e-6,
        raise_on_error=False,
    )

def compute_real2(x0_val,y0_val, a_val, b_val, static):
    last_x_np = np.zeros((1,))
    accrued_x_np = np.zeros((1,))
    accrued_temp = np.zeros((2,))

    stacked_x_np = np.zeros((5,1))
    stacked_x_new_np = np.zeros((5,1))
    stacked_temp_np = np.zeros((5,2))

    x_np = x0_val
    y_np = y0_val
    for i,j in zip(range(5), [1,0,1,0,0]):
        accrued_x_np += x_np
        stacked_x_np[i] = x_np
        # MAIN BODY LOOP
        x_np = x_np*a_val[i] + y_np[j]
        y_np = np.sin(y_np*b_val[i,j]*x_np/static)
        temp = x_np+y_np
        # MAIN BODY LOOP

        accrued_temp += temp
        stacked_x_new_np[i] = x_np
        stacked_temp_np[i] = temp
        last_x_np = x_np

    return last_x_np, accrued_x_np, accrued_temp, stacked_x_np, stacked_x_new_np, stacked_temp_np

def test_newloop2():
    rec = csdl.Recorder(inline = True)
    rec.start()

    a_val = np.array([0.1, 0.2, 0.3, 0.4, -0.1])
    b_val = np.arange(10).reshape(5,2)
    x0_val = np.array([2.0])
    y0_val = np.array([1.0, -1.0])
    static_val = np.array([0.5])

    x0 = csdl.Variable(name = 'x0', value = x0_val)
    y0 = csdl.Variable(name = 'y0', value = y0_val)
    a = csdl.Variable(name = 'a', value = a_val)
    b = csdl.Variable(name = 'b', value = b_val)
    static = csdl.Variable(name = 'static', value = static_val)

    with csdl.experimental.enter_loop(vals = [[0,1,2,3,4], [1,0,1,0,0]]) as loop_builder:
        i,j = loop_builder.get_loop_indices()
        x = loop_builder.initialize_feedback(x0)
        y = loop_builder.initialize_feedback(y0)
        x_new = x*a[i] + y[j]
        y_new = csdl.sin(y*b[i,j]*x_new/static)
        temp = x_new+y_new
        loop_builder.finalize_feedback(x, x_new)
        loop_builder.finalize_feedback(y, y_new)

    accrued_x = loop_builder.add_pure_accrue(x)
    accrued_temp = loop_builder.add_pure_accrue(temp)
    stacked_x = loop_builder.add_stack(x)
    stacked_x_new = loop_builder.add_stack(x_new)
    stacked_temp = loop_builder.add_stack(temp)
    last_x = loop_builder.add_output(x_new)
    loop_builder.finalize(add_all_outputs = False)

    last_x_np, accrued_x_np, accrued_temp_np, stacked_x_np, stacked_x_new_np, stacked_temp_np = compute_real2(x0_val, y0_val, a_val, b_val, static_val)
    assert np.allclose(accrued_temp_np, accrued_temp.value)
    assert np.allclose(last_x_np, last_x.value)
    assert np.allclose(accrued_x_np, accrued_x.value)
    assert np.allclose(stacked_x_np, stacked_x.value)
    assert np.allclose(stacked_x_new_np, stacked_x_new.value)
    assert np.allclose(stacked_temp_np, stacked_temp.value)

    x0_val = np.array([3.0])
    a_val = np.array([0.2, 0.3, 0.6, 0.5, -0.2])
    x0.value = x0_val
    a.value = a_val
    rec.execute()
    last_x_np, accrued_x_np, accrued_temp_np, stacked_x_np, stacked_x_new_np, stacked_temp_np = compute_real2(x0_val, y0_val, a_val, b_val, static_val)
    assert np.allclose(accrued_temp_np, accrued_temp.value)
    assert np.allclose(last_x_np, last_x.value)
    assert np.allclose(accrued_x_np, accrued_x.value)
    assert np.allclose(stacked_x_np, stacked_x.value)
    assert np.allclose(stacked_x_new_np, stacked_x_new.value)
    assert np.allclose(stacked_temp_np, stacked_temp.value)

    jax_interface = csdl.jax.create_jax_interface(inputs = [x0, a], outputs = [last_x, accrued_x, accrued_temp, stacked_x, stacked_x_new, stacked_temp])
    outputs = jax_interface({x0: x0_val, y0: y0_val,a: a_val, b: b_val, static: static_val})
    assert np.allclose(accrued_temp_np, outputs[accrued_temp])
    assert np.allclose(last_x_np, outputs[last_x])
    assert np.allclose(accrued_x_np, outputs[accrued_x])
    assert np.allclose(stacked_x_np, outputs[stacked_x])
    assert np.allclose(stacked_x_new_np, outputs[stacked_x_new])
    assert np.allclose(stacked_temp_np, outputs[stacked_temp])

    outs = csdl.derivative_utils.verify_derivatives(
        ofs = [last_x, accrued_x, accrued_temp, stacked_x, stacked_x_new, stacked_temp],
        wrts = [x0, a, y0, b, static],
        step_size=1e-9,
        raise_on_error=False,
    )

def compute_real3(x0_val,y0_val, a_val, b_val, static):
    last_x_np = np.zeros((1,))
    accrued_x_np = np.zeros((1,))
    accrued_temp = np.zeros((2,))

    stacked_x_np = np.zeros((5,1))
    stacked_x_new_np = np.zeros((5,1))
    stacked_temp_np = np.zeros((5,2))

    x_np = x0_val
    y_np = y0_val
    for i,j in zip(range(5), [1,0,1,0,0]):
        accrued_x_np += x_np
        stacked_x_np[i] = x_np
        # MAIN BODY LOOP
        x_np = x_np*a_val[i] + y_np[j]
        y_np = np.sin(y_np*b_val[i,j]*x_np/static)

        z = y_np
        for k in range(3):
            z = z + (k+i)/30.0
        y_np = z
        temp = x_np+y_np
        # MAIN BODY LOOP

        accrued_temp += temp
        stacked_x_new_np[i] = x_np
        stacked_temp_np[i] = temp
        last_x_np = x_np

    return last_x_np, accrued_x_np, accrued_temp, stacked_x_np, stacked_x_new_np, stacked_temp_np

def test_newloop3():
    rec = csdl.Recorder(inline = True)
    rec.start()

    a_val = np.array([0.1, 0.2, 0.3, 0.4, -0.1])
    b_val = np.arange(10).reshape(5,2)
    x0_val = np.array([2.0])
    y0_val = np.array([1.0, -1.0])
    static_val = np.array([0.5])

    x0 = csdl.Variable(name = 'x0', value = x0_val)
    y0 = csdl.Variable(name = 'y0', value = y0_val)
    a = csdl.Variable(name = 'a', value = a_val)
    b = csdl.Variable(name = 'b', value = b_val)
    static = csdl.Variable(name = 'static', value = static_val)

    x = x0
    y = y0
    with csdl.experimental.enter_loop(vals = [[0,1,2,3,4], [1,0,1,0,0]]) as loop_builder:
        i,j = loop_builder.get_loop_indices()
        x = loop_builder.initialize_feedback(x)
        y = loop_builder.initialize_feedback(y)
        x_new = x*a[i] + y[j]
        y_new = csdl.sin(y*b[i,j]*x_new/static)

        with csdl.experimental.enter_loop(3) as loop_builder2:
            k = loop_builder2.get_loop_indices()
            z = loop_builder2.initialize_feedback(y_new)
            z_new = z + (k+i)/30.0
            loop_builder2.finalize_feedback(z, z_new)
            y_new = loop_builder2.add_output(z_new)
        loop_builder2.finalize()

        temp = x_new+y_new
        loop_builder.finalize_feedback(x, x_new)
        loop_builder.finalize_feedback(y, y_new)

    accrued_x = loop_builder.add_pure_accrue(x)
    accrued_temp = loop_builder.add_pure_accrue(temp)
    stacked_x = loop_builder.add_stack(x)
    stacked_x_new = loop_builder.add_stack(x_new)
    stacked_temp = loop_builder.add_stack(temp)
    last_x = loop_builder.add_output(x_new)
    loop_builder.finalize(add_all_outputs = False)

    last_x_np, accrued_x_np, accrued_temp_np, stacked_x_np, stacked_x_new_np, stacked_temp_np = compute_real3(x0_val, y0_val, a_val, b_val, static_val)
    assert np.allclose(accrued_temp_np, accrued_temp.value)
    assert np.allclose(last_x_np, last_x.value)
    assert np.allclose(accrued_x_np, accrued_x.value)
    assert np.allclose(stacked_x_np, stacked_x.value)
    assert np.allclose(stacked_x_new_np, stacked_x_new.value)
    assert np.allclose(stacked_temp_np, stacked_temp.value)

    x0_val = np.array([3.0])
    a_val = np.array([0.2, 0.3, 0.6, 0.5, -0.2])
    x0.value = x0_val
    a.value = a_val
    rec.execute()
    last_x_np, accrued_x_np, accrued_temp_np, stacked_x_np, stacked_x_new_np, stacked_temp_np = compute_real3(x0_val, y0_val, a_val, b_val, static_val)
    assert np.allclose(accrued_temp_np, accrued_temp.value)
    assert np.allclose(last_x_np, last_x.value)
    assert np.allclose(accrued_x_np, accrued_x.value)
    assert np.allclose(stacked_x_np, stacked_x.value)
    assert np.allclose(stacked_x_new_np, stacked_x_new.value)
    assert np.allclose(stacked_temp_np, stacked_temp.value)

    jax_interface = csdl.jax.create_jax_interface(inputs = [x0, a], outputs = [last_x, accrued_x, accrued_temp, stacked_x, stacked_x_new, stacked_temp])
    outputs = jax_interface({x0: x0_val, a: a_val})
    assert np.allclose(accrued_temp_np, outputs[accrued_temp])
    assert np.allclose(last_x_np, outputs[last_x])
    assert np.allclose(accrued_x_np, outputs[accrued_x])
    assert np.allclose(stacked_x_np, outputs[stacked_x])
    assert np.allclose(stacked_x_new_np, outputs[stacked_x_new])
    assert np.allclose(stacked_temp_np, outputs[stacked_temp])

    outs = csdl.derivative_utils.verify_derivatives(
        ofs = [last_x, accrued_x, accrued_temp, stacked_x, stacked_x_new, stacked_temp],
        wrts = [x0, a, y0, b, static],
        step_size=1e-9,
        raise_on_error=False,
    )

if __name__ == '__main__':
    # test_build_vals()
    # test_loop_indices()
    test_newloop()
    # test_newloop2()
    # test_newloop3()