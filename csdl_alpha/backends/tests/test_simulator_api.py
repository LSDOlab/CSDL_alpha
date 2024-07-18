import pytest
import numpy as np

def get_sample_rec(shape = None):

    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()

    import csdl_alpha as csdl
    recorder = csdl.Recorder()
    recorder.start()

    x = csdl.Variable(name = "x", value = 2.0, shape = shape)
    y = csdl.Variable(name = "y", value = 3.0, shape = shape)

    z = x + y
    z.add_name("z")

    w = x*y
    w.add_name("w")
    recorder.stop()

    return (x,y), (z,w), recorder

def test_simulator_base_errors():
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()

    from csdl_alpha.backends.simulator import SimulatorBase

    with pytest.raises(TypeError) as e_info:
        simulator = SimulatorBase(recorder = 3)
    
    ins, outs, rec = get_sample_rec()

    sim = SimulatorBase(recorder = rec)

    with pytest.raises(NotImplementedError) as e_info:
        sim.run()

    with pytest.raises(NotImplementedError) as e_info:
        sim.run_forward()

    with pytest.raises(NotImplementedError) as e_info:
        sim.compute_optimization_derivatives()

def test_simulator1():
    from csdl_alpha.experimental import JaxSimulator
    from csdl_alpha.experimental import PySimulator

    for Simulator in [JaxSimulator, PySimulator]:
        from csdl_alpha.utils.hard_reload import hard_reload
        hard_reload()
        

        import csdl_alpha as csdl
        
        ins, outs, rec = get_sample_rec()

        for input_var in ins:
            input_var.set_as_design_variable()
        for output_var in outs:
            output_var.set_as_constraint()
        sim = Simulator(recorder = rec)

        # ITER 1:
        o, c = sim.run_forward()
        assert o is None
        np.testing.assert_almost_equal(c, np.array([5.0, 6.0]))

        g, j = sim.compute_optimization_derivatives()
        assert g is None
        np.testing.assert_almost_equal(j, np.array([[1.0, 1.0],[3.0, 2.0]]))

        # ITER 2:
        sim.update_design_variables(np.array([4.0, 5.0]))
        o, c = sim.run_forward()
        assert o is None
        np.testing.assert_almost_equal(c, np.array([9.0, 20.0]))
        
        g, j = sim.compute_optimization_derivatives()
        assert g is None
        np.testing.assert_almost_equal(j, np.array([[1.0, 1.0],[5.0, 4.0]]))

        g, j = sim.compute_optimization_derivatives(use_finite_difference = True)
        assert g is None
        np.testing.assert_almost_equal(j, np.array([[1.0, 1.0],[5.0, 4.0]]))

        sim.check_optimization_derivatives(raise_on_error = True)

def test_simulator2():
    from csdl_alpha.experimental import JaxSimulator
    from csdl_alpha.experimental import PySimulator

    for Simulator in [JaxSimulator, PySimulator]:
        from csdl_alpha.utils.hard_reload import hard_reload
        hard_reload()
        
        from csdl_alpha.experimental import PySimulator
        import csdl_alpha as csdl
        
        ins, outs, rec = get_sample_rec(shape = (2,2))

        rec.start()
        for input_var in ins:
            input_var.set_as_design_variable()
        outs[0].set_as_constraint()
        outs[1][0,0].set_as_objective()
        rec.stop()

        sim = Simulator(recorder = rec)

        # ITER 1:
        o, c = sim.run_forward()
        np.testing.assert_almost_equal(c, np.array([5.0, 5.0, 5.0, 5.0]))
        np.testing.assert_almost_equal(o, np.array([6.0]))

        g, j = sim.compute_optimization_derivatives()
        np.testing.assert_almost_equal(g, np.hstack((3*np.eye(4), 2*np.eye(4)))[0].reshape(1,-1))
        np.testing.assert_almost_equal(j, np.hstack((np.eye(4), np.eye(4))))

        # ITER 2:
        new_x0 = np.arange(4).reshape((2,2))
        new_x1 = np.arange(4).reshape((2,2))+6
        new_x_concat = np.hstack((new_x0.flatten(), new_x1.flatten()))
        sim.update_design_variables(new_x_concat)
        o, c= sim.run_forward()
        np.testing.assert_almost_equal(c, (new_x0.flatten()+new_x1.flatten()))
        np.testing.assert_almost_equal(o, (new_x0.flatten()*new_x1.flatten())[0])

        g, j = sim.compute_optimization_derivatives()
        np.testing.assert_almost_equal(g, np.hstack((np.diagflat(new_x1), np.diagflat(new_x0)))[0].reshape(1,-1))
        np.testing.assert_almost_equal(j, np.hstack((np.eye(4), np.eye(4))))

        g, j = sim.compute_optimization_derivatives(use_finite_difference = True)
        np.testing.assert_almost_equal(g, np.hstack((np.diagflat(new_x1), np.diagflat(new_x0)))[0].reshape(1,-1))
        np.testing.assert_almost_equal(j, np.hstack((np.eye(4), np.eye(4))))

        sim.check_optimization_derivatives(raise_on_error = True)

if __name__ == "__main__":
    # test_simulator_base_errors()
    # test_simulator1()
    test_simulator2()