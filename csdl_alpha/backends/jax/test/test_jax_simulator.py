import pytest

def build_model():
    import numpy as np
    import csdl_alpha as csdl

    # Inputs
    x1 = csdl.Variable(name='x1', shape=(1,), value=np.array([1.0]))
    x2 = csdl.Variable(name='x2', shape=(1,), value=np.array([2.0]))
    x3 = csdl.Variable(name='x3', shape=(3,), value=np.array([3.0]))
    x4 = csdl.Variable(name='x4', shape=(1,), value=np.array([4.0]))
    x5 = csdl.Variable(name='x5', shape=(4,4), value=np.array([5.0]))
    
    # Outputs
    inter = x1 + x2
    z1 = 3*inter
    z2 = inter**x4
    z3 = z1**2.0*x3[0]
    z4 = csdl.sum(x3) + x4
    z5 = z1*z4

    # numpy check function
    def func(x1_val, x2_val, x3_val, x4_val, x5_val):
        inter = x1_val + x2_val
        z1 = 3*inter
        z2 = inter**x4_val
        z3 = z1**2.0*x3_val[0]
        z4 = np.sum(x3_val) + x4_val
        z5 = z1*z4
        return z1, z2, z3, z4, z5

    return (x1, x2, x3, x4, x5), (z1, z2, z3, z4, z5), func

def test_jsimulator_errors():
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()
    import numpy as np
    import csdl_alpha as csdl

    recorder = csdl.Recorder()
    recorder.start()
    inputs, outputs, fwd_func = build_model()
    recorder.stop()

    JaxSimulator = csdl.experimental.JaxSimulator
    
    # Check argument types
    with pytest.raises(TypeError) as e_info:
        sim = JaxSimulator(
            recorder=recorder, 
            additional_inputs='a', 
            additional_outputs=outputs,
        )

    with pytest.raises(TypeError) as e_info:
        sim = JaxSimulator(
            recorder=recorder, 
            additional_inputs=inputs, 
            additional_outputs='a',
        )

    with pytest.raises(TypeError) as e_info:
        sim = JaxSimulator(
            recorder=recorder, 
            additional_inputs=inputs, 
            additional_outputs=outputs,
            output_saved='a',
        )
    
    with pytest.raises(TypeError) as e_info:
        sim = JaxSimulator(
            recorder=recorder, 
            additional_inputs=inputs, 
            additional_outputs=outputs,
            output_saved=True,
            gpu='a',
        )

    # Check to make sure outputs are not inputs
    with pytest.raises(ValueError) as e_info:
        sim = JaxSimulator(
            recorder=recorder, 
            additional_inputs=outputs, 
            additional_outputs=outputs,
        )

    sim = JaxSimulator(
        recorder=recorder, 
        additional_inputs=inputs, 
        additional_outputs=outputs,
    )

def test_jsimulator():
    """
    Test all inputs and all outputs
    """
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()
    import numpy as np
    import csdl_alpha as csdl

    recorder = csdl.Recorder()
    recorder.start()
    inputs, outputs, fwd_func = build_model()
    recorder.stop()

    JaxSimulator = csdl.experimental.JaxSimulator
    jax_sim = JaxSimulator(
        recorder=recorder, 
        additional_inputs=inputs, 
        additional_outputs=outputs,
    )
    jax_sim.run()
    jax_sim.compute_totals()
    jax_sim.check_totals(raise_on_error=True)
    output_vals = fwd_func(*[input.value for input in inputs])
    for i, output in enumerate(outputs):
        assert np.allclose(output_vals[i], jax_sim[output])
    
    new_vals = []
    for i in range(5):
        val = i*np.ones(inputs[i].shape)
        jax_sim[inputs[i]] = val
        new_vals.append(val)

    jax_sim.run()
    jax_sim.compute_totals()
    jax_sim.check_totals(raise_on_error=True)

    output_vals = fwd_func(*[new_val for new_val in new_vals])
    for i, output in enumerate(outputs):
        assert np.allclose(output_vals[i], jax_sim[output])

    # Check to optimization functions cannot be ran without design variables/constraints etc
    with pytest.raises(ValueError) as e_info:
        jax_sim.run_forward()
    with pytest.raises(ValueError) as e_info:
        jax_sim.compute_optimization_derivatives()
        
def test_jsimulator_constants():
    """
    Test some inputs and some outputs
    """
    # Build model
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()
    import numpy as np
    import csdl_alpha as csdl

    recorder = csdl.Recorder()
    recorder.start()
    all_inputs, all_outputs, fwd_func = build_model()
    recorder.stop()
    
    # Set inputs and create simulator
    inputs = all_inputs[0:3]
    outputs = all_outputs[0:3]
    JaxSimulator = csdl.experimental.JaxSimulator
    jax_sim = JaxSimulator(
        recorder=recorder, 
        additional_inputs=inputs, 
        additional_outputs=outputs,
    )

    # Check initial values
    jax_sim.run()
    jax_sim.compute_totals()
    jax_sim.check_totals(raise_on_error=True)
    output_vals = fwd_func(*[input.value for input in all_inputs])
    for i, output in enumerate(outputs):
        assert np.allclose(output_vals[i], jax_sim[output])
    
    # Set new values
    new_vals = []
    for i in range((len(inputs))):
        val = i**2.0**np.ones(inputs[i].shape)
        jax_sim[inputs[i]] = val
        new_vals.append(val)

    # Check new values
    jax_sim.run()
    jax_sim.compute_totals()
    jax_sim.check_totals(raise_on_error=True)

    output_vals = fwd_func(*([new_val for new_val in new_vals]+[input.value for input in all_inputs[len(inputs):]]))
    for i, output in enumerate(outputs):
        assert np.allclose(output_vals[i], jax_sim[output])

    # Check to make sure non-outputs/inputs cannot be setted/getted
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_inputs[3]] = 1.0
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_inputs[4]] = 1.0
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_outputs[3]]
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_outputs[4]]

    # Check to make sure outputs cannot be setted
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_outputs[1]] = 1.0
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_outputs[2]] = 2.0

    # Check to optimization functions cannot be ran without design variables/constraints etc
    with pytest.raises(ValueError) as e_info:
        jax_sim.run_forward()
    with pytest.raises(ValueError) as e_info:
        jax_sim.compute_optimization_derivatives()

def test_jsimulator_optimization():
    """
    Test some inputs and some outputs
    """
    # Build model
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()
    import numpy as np
    import csdl_alpha as csdl

    recorder = csdl.Recorder()
    recorder.start()
    all_inputs, all_outputs, fwd_func = build_model()
    recorder.stop()
    # Set inputs and create simulator
    inputs = all_inputs[0:3]
    additional_inputs = inputs[0:2]
    inputs[2].set_as_design_variable()

    outputs = all_outputs[0:3]
    additional_outputs = outputs[0:2]
    outputs[2].set_as_objective()
    
    JaxSimulator = csdl.experimental.JaxSimulator
    jax_sim = JaxSimulator(
        recorder=recorder, 
        additional_inputs=additional_inputs, 
        additional_outputs=outputs,
    )

    # Check initial values
    jax_sim.run_forward()
    old_grad, _ = jax_sim.compute_optimization_derivatives()
    jax_sim.compute_totals()
    jax_sim.check_totals(raise_on_error=True)
    output_vals = fwd_func(*[input.value for input in all_inputs])
    for i, output in enumerate(outputs):
        assert np.allclose(output_vals[i], jax_sim[output])
    
    # Set new values
    new_vals = []
    for i in range((len(inputs))):
        val = i**2.0**np.ones(inputs[i].shape)
        jax_sim[inputs[i]] = val
        new_vals.append(val)

    # Check new values of optimization only. Additional outputs should not be updated
    jax_sim.run_forward()
    new_grad, _ = jax_sim.compute_optimization_derivatives()
    assert not np.allclose(old_grad, new_grad)

    output_vals = fwd_func(*([new_val for new_val in new_vals]+[input.value for input in all_inputs[len(inputs):]]))
    assert not np.allclose(output_vals[0], jax_sim[outputs[0]])
    assert not np.allclose(output_vals[1], jax_sim[outputs[1]])
    assert np.allclose(output_vals[2], jax_sim[outputs[2]]) # only objective is updated

    jax_sim.run()
    assert np.allclose(output_vals[0], jax_sim[outputs[0]])
    assert np.allclose(output_vals[1], jax_sim[outputs[1]]) # other outputs are updated now
    assert np.allclose(output_vals[2], jax_sim[outputs[2]]) # other outputs are updated now

    # When only the design variable is updated, the function should not be recompiled
    jax_sim[inputs[2]] = 20.0*np.ones(inputs[2].shape)
    assert jax_sim.run_forward_func is not None
    assert jax_sim.run_func is not None
    assert jax_sim.totals_derivs is not None
    assert jax_sim.opt_derivs_func is not None
    jax_sim.run_forward()

    # When a non design variable input is updated, the optimization function should be recompiled
    jax_sim[inputs[1]] = 20.0*np.ones(inputs[1].shape)
    assert jax_sim.run_forward_func is None
    assert jax_sim.run_func is not None
    assert jax_sim.totals_derivs is not None
    assert jax_sim.opt_derivs_func is None

    # Check to make sure non-outputs/inputs cannot be setted/getted
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_inputs[3]] = 1.0
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_inputs[4]] = 1.0
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_outputs[3]]
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_outputs[4]]

    # Check to make sure outputs cannot be setted
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_outputs[1]] = 1.0
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_outputs[2]] = 2.0
    jax_sim.check_optimization_derivatives(raise_on_error = True)


def test_jsimulator_optimization_finite_difference():
    """
    Test some inputs and some outputs
    """
    # Build model
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()
    import numpy as np
    import csdl_alpha as csdl

    recorder = csdl.Recorder()
    recorder.start()
    all_inputs, all_outputs, fwd_func = build_model()
    # Set inputs and create simulator
    inputs = all_inputs[0:3]
    additional_inputs = inputs[0:2]
    inputs[2].set_as_design_variable()

    outputs = all_outputs[0:3]
    outputs = list(outputs)
    outputs[2].set_as_objective()
    outputs[1] = outputs[1]
    outputs[1].set_as_constraint()
    outputs = tuple(outputs)
    additional_outputs = outputs[0:2]
    recorder.stop()
    
    JaxSimulator = csdl.experimental.JaxSimulator
    jax_sim = JaxSimulator(
        recorder=recorder, 
        additional_inputs=additional_inputs, 
        additional_outputs=outputs,
    )

    # Check initial values
    jax_sim.run_forward()
    old_grad, old_jac = jax_sim.compute_optimization_derivatives(use_finite_difference=True)
    jax_sim.compute_totals()
    jax_sim.check_totals(raise_on_error=True)
    output_vals = fwd_func(*[input.value for input in all_inputs])
    for i, output in enumerate(outputs):
        assert np.allclose(output_vals[i], jax_sim[output])
    
    # Set new values
    new_vals = []
    for i in range((len(inputs))):
        val = i**2.0**np.ones(inputs[i].shape)
        jax_sim[inputs[i]] = val
        new_vals.append(val)

    # Check new values of optimization only. Additional outputs should not be updated
    jax_sim.run_forward()
    new_grad, new_jac = jax_sim.compute_optimization_derivatives(use_finite_difference=True)
    assert not np.allclose(old_grad, new_grad)
    new_grad_a, new_jac_a = jax_sim.compute_optimization_derivatives(use_finite_difference=False)
    assert np.allclose(new_grad, new_grad_a)
    assert np.allclose(new_jac, new_jac_a)

    output_vals = fwd_func(*([new_val for new_val in new_vals]+[input.value for input in all_inputs[len(inputs):]]))
    assert not np.allclose(output_vals[0], jax_sim[outputs[0]])
    assert np.allclose(output_vals[1], jax_sim[outputs[1]])
    assert np.allclose(output_vals[2], jax_sim[outputs[2]]) # only objective is updated

    jax_sim.run()
    assert np.allclose(output_vals[0], jax_sim[outputs[0]])
    assert np.allclose(output_vals[1], jax_sim[outputs[1]]) # other outputs are updated now
    assert np.allclose(output_vals[2], jax_sim[outputs[2]]) # other outputs are updated now

    # When only the design variable is updated, the function should not be recompiled
    jax_sim[inputs[2]] = 20.0*np.ones(inputs[2].shape)
    assert jax_sim.run_forward_func is not None
    assert jax_sim.run_func is not None
    assert jax_sim.totals_derivs is not None
    assert jax_sim.opt_derivs_func is not None
    jax_sim.run_forward()

    # When a non design variable input is updated, the optimization function should be recompiled
    jax_sim[inputs[1]] = 20.0*np.ones(inputs[1].shape)
    assert jax_sim.run_forward_func is None
    assert jax_sim.run_func is not None
    assert jax_sim.totals_derivs is not None
    assert jax_sim.opt_derivs_func is None

    # Check to make sure non-outputs/inputs cannot be setted/getted
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_inputs[3]] = 1.0
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_inputs[4]] = 1.0
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_outputs[3]]
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_outputs[4]]

    # Check to make sure outputs cannot be setted
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_outputs[1]] = 1.0
    with pytest.raises(KeyError) as e_info:
        jax_sim[all_outputs[2]] = 2.0

    jax_sim.check_optimization_derivatives(raise_on_error = True)

def test_save_hd5f(tmp_path):
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()
    import os
    os.chdir(tmp_path)

    import csdl_alpha as csdl

    recorder = csdl.Recorder()
    recorder.start()
    all_inputs, all_outputs, fwd_func = build_model()
    recorder.stop()
    
    # Set inputs and create simulator
    inputs = all_inputs[0:3]
    additional_inputs = inputs[0:2]
    inputs[2].set_as_design_variable()

    outputs = all_outputs[0:3]
    outputs[2].set_as_objective()
    outputs[0].save()
    outputs[1].save()

    jaxsim = csdl.experimental.JaxSimulator(
        recorder=recorder, 
        output_saved=True,
        additional_inputs=additional_inputs,
    )
    
    jaxsim.run()
    jaxsim.save_external('test', 1)

if __name__ == '__main__':
    # test_jsimulator_errors()
    # test_jsimulator()
    # test_jsimulator_constants()
    # test_jsimulator_optimization()
    test_jsimulator_optimization_finite_difference()
