import csdl_alpha.utils.test_utils as csdl_tests
from csdl_alpha.src.graph.variable import Variable 

class TestVariable(csdl_tests.CSDLTest):
    def test_docstrings(self):
        self.docstest(Variable.flatten) 
        self.docstest(Variable.reshape)
        self.docstest(Variable.T)
        self.docstest(Variable.__getitem__)
        self.docstest(Variable.set)

    def test_int_float(self):
        self.prep()
        import csdl_alpha as csdl
        import numpy as np

        a = csdl.Variable(value=2, name='a')
        b = csdl.Variable(value=3.0, name='b')

        assert isinstance(a.value[0], np.float64)
        assert isinstance(b.value[0], np.float64)

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
    