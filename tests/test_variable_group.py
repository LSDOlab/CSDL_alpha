import pytest
from typing import Union
from dataclasses import dataclass
from csdl_alpha.src.variable_group import VariableGroup
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.test_utils import CSDLTest, TestingPair
import numpy as np


class TestVariableGroup(CSDLTest):
    def test_custom_variable_group(self):
        self.prep()

        @dataclass
        class CustomVariableGroup(VariableGroup):
            a : Union[Variable, int, float]
            b : Variable
            def define_checks(self):
                self.add_check('a', shape=(1,), variablize=True)
                self.add_check('b', type=Variable, shape=(1,))

        # test basics
        a = 1
        b = Variable(shape=(1,), value=1)
        vg = CustomVariableGroup(a, b)
        assert isinstance(vg.a, Variable)
        assert vg.a.value == np.array([1])
        assert vg.b.value == np.array([1])

        # tests saving
        vg.save()
        assert vg.b._save == True
        assert vg.a._save == True

        # test adding tag
        vg.add_tag('test')
        assert 'test' in vg.a.tags
        assert 'test' in vg.b.tags

        # test error for declaring parameters twice
        with pytest.raises(ValueError):
            vg.add_check('a', shape=(1,), variablize=True)

        # test error for declaring parameters that don't exist
        with pytest.raises(ValueError):
            vg.add_check('c', shape=(1,), variablize=True)

        # test error for setting variable of wrong type
        with pytest.raises(ValueError):
            vg.b = 'banana'

        # test error for setting variable of wrong shape
        with pytest.raises(ValueError):
            vg.b = Variable(shape=(2,))

        # test error when removing a variable
        with pytest.raises(ValueError):
            del vg.b
            vg.check()

    def test_non_dataclass_vg(self):
        self.prep()

        class CustomVariableGroup(VariableGroup):
            a : Union[Variable, int, float]
            b : Variable
            def define_checks(self):
                self.add_check('a', shape=(1,), variablize=True)
                self.add_check('b', type=Variable, shape=(1,))

        with pytest.raises(TypeError):
            CustomVariableGroup()        