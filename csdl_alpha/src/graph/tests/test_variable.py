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
    