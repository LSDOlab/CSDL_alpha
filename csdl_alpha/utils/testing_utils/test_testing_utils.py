import pytest
from csdl_alpha.utils.testing_utils.csdl_test import CSDLTest, TestingPair
import numpy as np
import csdl_alpha as csdl


def test_tp_wrong_val():
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()
    rec = csdl.build_new_recorder()
    rec.start()
    y = np.ones((5,5))*5.5
    x = csdl.Variable(name = 'x', value = y+0.01)
    tp = TestingPair(x,y)
    with pytest.raises(AssertionError) as e_info:
        tp.compare(0)

def test_tp_wrong_shape():
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()
    rec = csdl.build_new_recorder()
    rec.start()
    y = np.ones((5,5))*5.5
    x = csdl.Variable(name = 'x', value = 5.5)
    tp = TestingPair(x,y)
    with pytest.raises(AssertionError) as e_info:
        tp.compare(0)

def test_tp_decimal():
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()
    rec = csdl.build_new_recorder()
    rec.start()
    y = np.ones((5,5))*5.5
    x = csdl.Variable(name = 'x', value = y+1e-6)
    tp = TestingPair(x,y, decimal = 8)
    with pytest.raises(AssertionError) as e_info:
        tp.compare(0)

def test_tp_1():
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()

    x = np.ones((5,5))
    with pytest.raises(TypeError) as e_info:
        tp = TestingPair(x,x)

def test_tp_2():
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()

    # TODO: RAISE error if recorder is not started before creating a variable
    rec = csdl.build_new_recorder()
    rec.start()
    x = csdl.Variable(name = 'x', value = 1.5)
    with pytest.raises(TypeError) as e_info:
        tp = TestingPair(x,x)