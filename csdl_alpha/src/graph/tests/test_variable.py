import csdl_alpha.utils.test_utils as csdl_tests
from csdl_alpha.src.graph.variable import Variable 

class TestVariable(csdl_tests.CSDLTest):
    def test_docstrings(self):
        self.docstest(Variable.flatten) 
        self.docstest(Variable.reshape)
        self.docstest(Variable.T)
        self.docstest(Variable.__getitem__)
        self.docstest(Variable.set)
    