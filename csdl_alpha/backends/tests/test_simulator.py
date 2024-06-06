from csdl_alpha.backends.tests.test_simulator_api import get_sample_rec
import numpy as np

def test_optimization_determine():
    """
    Test the determine_if_optimization function
    """
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()

    from csdl_alpha.backends.simulator import determine_if_optimization

    ins, outs, rec = get_sample_rec()
    assert not determine_if_optimization(rec)

    ins, outs, rec = get_sample_rec()
    for output_var in outs:
        output_var.set_as_constraint()
    assert not determine_if_optimization(rec)

    ins, outs, rec = get_sample_rec()
    for input_var in ins:
        input_var.set_as_design_variable()
    assert not determine_if_optimization(rec)

    ins, outs, rec = get_sample_rec()
    for input_var in ins:
        input_var.set_as_design_variable()
    for output_var in outs:
        output_var.set_as_constraint()
    assert determine_if_optimization(rec)

def test_optimization_meta():
    """
    Test the determine_if_optimization function
    """
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()

    ins, outs, rec = get_sample_rec()
    for in_var in ins:
        in_var.set_as_design_variable()
    outs[0].set_as_constraint()
    outs[1].set_as_objective()

    dv_size = sum(in_var.size for in_var in ins)
    c_size = outs[0].size
    o_size = outs[1].size

    from csdl_alpha.backends.simulator import SimulatorBase
    sim = SimulatorBase(recorder = rec)
    dmeta, cmeta, ometa = sim.get_optimization_metadata()
    assert len(dmeta) == 4
    assert len(cmeta) == 3
    assert len(ometa) == 1

    np.testing.assert_almost_equal(ometa, np.ones((o_size,)))
    np.testing.assert_almost_equal(cmeta[0], np.ones((c_size,)))
    np.testing.assert_almost_equal(dmeta[0], np.ones((dv_size,)))
    
    np.testing.assert_almost_equal(cmeta[1], -np.inf*np.ones((c_size,)))
    np.testing.assert_almost_equal(dmeta[1], -np.inf*np.ones((dv_size,)))

    np.testing.assert_almost_equal(cmeta[2], np.inf*np.ones((c_size,)))
    np.testing.assert_almost_equal(dmeta[2], np.inf*np.ones((dv_size,)))

def test_optimization_meta_1():
    """
    Test the determine_if_optimization function
    """
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()

    ins, outs, rec = get_sample_rec(shape = (3,3))
    ins[0].set_as_design_variable(upper = 0.0, scaler = 2.0)
    ins[1].set_as_design_variable(lower = np.arange(9).reshape(3,3), upper = 10.0, scaler = 1e3*np.arange(9).reshape(3,3))

    rec.start()
    outs[0].set_as_constraint()
    outs[1][0,0].set_as_objective()
    rec.stop()

    dv_size = sum(in_var.size for in_var in ins)
    c_size = outs[0].size
    o_size = 1

    from csdl_alpha.backends.simulator import SimulatorBase
    sim = SimulatorBase(recorder = rec)
    dmeta, cmeta, ometa = sim.get_optimization_metadata()
    assert len(dmeta) == 4
    assert len(cmeta) == 3
    assert isinstance(ometa, np.ndarray)

    # print('scaler:', dmeta[0])
    # print('lower:', dmeta[1])
    # print('upper:', dmeta[2])
    # print('value:', dmeta[3])

    # scaler:
    np.testing.assert_almost_equal(ometa, np.ones((o_size,)))
    np.testing.assert_almost_equal(cmeta[0], np.ones((c_size,)))
    np.testing.assert_almost_equal(dmeta[0][:9], 2.0*np.ones((9,)))
    np.testing.assert_almost_equal(dmeta[0][9:], 1e3*np.arange(9))
    
    # lower:
    np.testing.assert_almost_equal(cmeta[1], -np.inf*np.ones((c_size,)))
    np.testing.assert_almost_equal(dmeta[1][:9], -np.inf*np.ones((9,)))
    np.testing.assert_almost_equal(dmeta[1][9:], np.arange(9))

    # upper:
    np.testing.assert_almost_equal(cmeta[2], np.inf*np.ones((c_size,)))
    np.testing.assert_almost_equal(dmeta[2][:9], np.zeros((9,)))
    np.testing.assert_almost_equal(dmeta[2][9:], 10.0+np.zeros((9,)))

def test_optimization_meta_2():
    """
    Test the determine_if_optimization function
    """
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()

    ins, outs, rec = get_sample_rec(shape = (3,3))
    ins[0].set_as_design_variable(upper = 0.0, scaler = 2.0)
    ins[1].set_as_design_variable(lower = np.arange(9).reshape(3,3), upper = 10.0, scaler = 1e3*np.arange(9).reshape(3,3))

    rec.start()
    outs[1][0,0].set_as_objective()
    rec.stop()

    dv_size = sum(in_var.size for in_var in ins)
    c_size = outs[0].size
    o_size = 1

    from csdl_alpha.backends.simulator import SimulatorBase
    sim = SimulatorBase(recorder = rec)
    dmeta, cmeta, ometa = sim.get_optimization_metadata()
    assert len(dmeta) == 4
    assert len(cmeta) == 3
    assert isinstance(ometa, np.ndarray)

    # print('scaler:', dmeta[0])
    # print('lower:', dmeta[1])
    # print('upper:', dmeta[2])
    # print('value:', dmeta[3])

    # scaler:
    np.testing.assert_almost_equal(ometa, np.ones((o_size,)))
    assert cmeta[0] is None
    np.testing.assert_almost_equal(dmeta[0][:9], 2.0*np.ones((9,)))
    np.testing.assert_almost_equal(dmeta[0][9:], 1e3*np.arange(9))
    
    # lower:
    assert cmeta[1] is None
    np.testing.assert_almost_equal(dmeta[1][:9], -np.inf*np.ones((9,)))
    np.testing.assert_almost_equal(dmeta[1][9:], np.arange(9))

    # upper:
    assert cmeta[2] is None
    np.testing.assert_almost_equal(dmeta[2][:9], np.zeros((9,)))
    np.testing.assert_almost_equal(dmeta[2][9:], 10.0+np.zeros((9,)))

if __name__ == "__main__":
    test_optimization_determine()
    test_optimization_meta()
    test_optimization_meta_1()
    test_optimization_meta_2()