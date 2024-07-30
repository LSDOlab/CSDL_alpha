from csdl_alpha.backends.jax.test.test_jax_simulator import build_model
import pytest

def test_pysimulator():
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

    py_sim = csdl.experimental.PySimulator(
        recorder=recorder, 
    )
    py_sim.run()
    totals = py_sim.compute_totals(outputs, inputs)
    py_sim.check_totals(outputs, inputs, raise_on_error=True)
    output_vals = fwd_func(*[input.value for input in inputs])
    for i, output in enumerate(outputs):
        assert np.allclose(output_vals[i], py_sim[output])

    new_vals = []
    for i in range(5):
        val = i*np.ones(inputs[i].shape)
        py_sim[inputs[i]] = val
        new_vals.append(val)

    py_sim.run()
    py_sim.compute_totals(outputs, inputs)
    py_sim.check_totals(outputs, inputs, raise_on_error=True)

    output_vals = fwd_func(*[new_val for new_val in new_vals])
    for i, output in enumerate(outputs):
        assert np.allclose(output_vals[i], py_sim[output])

    # Check to optimization functions cannot be ran without design variables/constraints etc
    with pytest.raises(ValueError) as e_info:
        py_sim.run_forward()
    with pytest.raises(ValueError) as e_info:
        py_sim.compute_optimization_derivatives()

    # Check to make sure outputs cannot be setted
    for out in outputs:
        with pytest.raises(ValueError) as e_info:
            py_sim[out] = 1.0

if __name__ == "__main__":
    test_pysimulator()