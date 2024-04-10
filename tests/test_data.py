import pytest
from csdl_alpha.utils.test_utils import CSDLTest

class TestData(CSDLTest):
    def test_data(self):
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.src.graph.variable import Variable
        import numpy as np

        a = Variable(value=2, name='a')
        a.add_tag('test')
        a.hierarchy = 1
        b = Variable(value=3)
        csdl.save_all_variables()

        dv = Variable(value=4)
        constraint = dv*2
        constraint.is_input = False
        objective = dv*3
        objective.is_input = False
        dv.set_as_design_variable()
        constraint.set_as_constraint()
        objective.set_as_objective()

        csdl.save_optimization_variables()

        csdl.inline_save('test_data')

        variables = csdl.import_h5py('test_data.hdf5', 'inline')
        assert variables['a'].value == a.value
        assert variables['a'].tags == a.tags
        assert variables['a'].hierarchy == a.hierarchy
        assert variables['a'].name == a.name

# if __name__ == '__main__':
#     import csdl_alpha as csdl
#     recorder = csdl.Recorder()
#     recorder.start()
#     vars = csdl.import_h5py('test_data.hdf5', 'inline')
#     print(vars)


        