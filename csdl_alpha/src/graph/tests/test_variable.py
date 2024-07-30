import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.src.graph.variable import Variable 

import pytest

class TestVariable(csdl_tests.CSDLTest):
    def test_docstrings(self):
        self.docstest(Variable.flatten) 
        self.docstest(Variable.reshape)
        self.docstest(Variable.T)
        self.docstest(Variable.__getitem__)
        self.docstest(Variable.set)
        self.docstest(Variable.expand)

    def test_int_float(self):
        self.prep()
        import csdl_alpha as csdl
        import numpy as np

        a = csdl.Variable(value=2, name='a')
        b = csdl.Variable(value=3.0, name='b')

        assert isinstance(a.value[0], np.float64)
        assert isinstance(b.value[0], np.float64)

    def test_np_matrix_convert(self):
        self.prep()
        import csdl_alpha as csdl
        import numpy as np

        a = csdl.Variable(value=np.matrix([[1.0, 2.0],[3.0, 4.0]]), name='a')
        assert not isinstance(a.value, np.matrix)

        a.set_value(np.matrix([[1.0, 2.0],[3.0, 4.0]]))
        assert not isinstance(a.value, np.matrix)

    def test_variable_inputs(self):
        import csdl_alpha as csdl
        import numpy as np
        self.prep()

        # standard variable creation - shape and value
        a = csdl.Variable((1,), value=1)
        assert a.shape == (1,)
        assert a.value == np.ones((1,))
        b = csdl.Variable((1,))
        assert b.shape == (1,)
        assert b.value == None
        b.set_value(1)
        assert b.value == np.ones((1,))
        c = csdl.Variable((1,), value=np.array([1]))
        assert c.shape == (1,)
        assert c.value == np.array([1])
        d = csdl.Variable((10,10), value=1)
        assert d.shape == (10,10)
        assert np.all(d.value == np.ones((10,10)))
    

    def test_variable_optimization(self):
        import csdl_alpha as csdl
        import numpy as np
        self.prep()

        c = csdl.Variable((2,), value=1)
        with pytest.raises(ValueError):
            c.set_as_objective()

        a = csdl.Variable((1,), value=1)
        a.set_as_objective()

        b = csdl.Variable((1,), value=1)
        with pytest.raises(ValueError):
            b.set_as_objective()

    def test_variable_value_property(self):
        import csdl_alpha as csdl
        import numpy as np
        self.prep()

        a = csdl.Variable((2,), value=1)
        
        with pytest.raises(TypeError):
            a.value = 'string'

        with pytest.raises(ValueError):
            a.value = np.ones((2,2))

        with pytest.raises(TypeError):
            a.value = csdl.Variable((2,), value=1)

        a.value = np.ones((2,), dtype=np.float32)
        assert a.value.dtype == np.float64

    def test_variable_value_inline_none(self):
        from csdl_alpha.utils.hard_reload import hard_reload
        hard_reload()

        import csdl_alpha as csdl

        # No error if inline == False
        recorder = csdl.Recorder()
        recorder.start()
        a = csdl.Variable((2,), value=None)
        recorder.stop()

        # No error if inline == True but variable doesn't get used
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        a = csdl.Variable((2,), value=None)
        recorder.stop()

        # Error if inline == True and variable gets used
        recorder = csdl.Recorder(inline=True)
        recorder.start()
        a = csdl.Variable((2,), value=None)
        with pytest.raises(ValueError) as e:
            b = a + 1
        assert "must have a value set when running in inline mode." in str(e.value)

        with pytest.raises(ValueError) as e:
            b = a*a
        assert "must have a value set when running in inline mode." in str(e.value)

        with pytest.raises(ValueError) as e:
            b = csdl.sum(a,a,a)
        assert "must have a value set when running in inline mode." in str(e.value)

        recorder.stop()