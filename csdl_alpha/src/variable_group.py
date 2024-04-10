from typing import Union
import warnings
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize
from dataclasses import dataclass, is_dataclass

# TODO: add freeze and whatnot
@dataclass
class VariableGroup:

    def __post_init__(self):
        if not is_dataclass(self):
            raise ValueError("VariableGroup must be a dataclass.")
        self._metadata = {}
        self.define()
        self.check()

    def __setattr__(self, name, value):
        if hasattr(self, '_metadata'):
            value = self._check_pamaeters(name, value)
        super().__setattr__(name, value)

    def _check_pamaeters(self, name, value):
        if name in self._metadata:
            if self._metadata[name]['variablize']:
                value = variablize(value)
            if self._metadata[name]['type'] is not None:
                if type(value) != self._metadata[name]['type']:
                    raise ValueError(f"Variable {name} must be of type {self._metadata[name]['type']}.")
            if self._metadata[name]['shape'] is not None:
                if value.shape != self._metadata[name]['shape']:
                    raise ValueError(f"Variable {name} must have shape {self._metadata[name]['shape']}.")
        return value

    def check(self):
        for key in self._metadata.keys():
            if not hasattr(self, key):
                raise ValueError(f"Variable {key} not found in the group.")
            val = getattr(self, key)
            setattr(self, key, self._check_pamaeters(key, val))

    def define(self):
        pass
    
    def declare_parameters(self, name, type=None, shape=None, variablize=False):
        """Declare parameters for a variable in the group.

        This method is used to declare parameters for a variable in the group. The parameters
        include the name, type, shape, and whether the variable should be variablized.

        Parameters
        ----------
        name : str
            The name of the variable.
        type : type, optional
            The type of the variable, by default None.
        shape : type, optional
            The shape of the variable, by default None.
        variablize : bool, optional
            Whether the variable should be turned into a CSDL variable, by default False.

        Raises
        ------
        ValueError
            If the variable with the given name is not found in the group.
        ValueError
            If parameters for the variable with the given name are already declared.
        """
        if not name in self.__annotations__:
            raise ValueError(f"Variable {name} not found in the group.")
        if name in self._metadata:
            raise ValueError(f"Parameters for variable {name} already declared.")
        
        self._metadata[name] = {'type': type, 'shape': shape, 'variablize': variablize}

    def add_tag(self, tag):
        """Adds a tag to all Variables in the group or subgroups.

        Parameters
        ----------
        tag : str
            Tag to add to the Variables.
        """
        for key, val in self.__dict__.items():
            if type(val) == Variable or type(val) == VariableGroup:
                val.add_tag(tag)

    def save(self):
        """saves any Variables in the group or subgroups.
        """
        for key, val in self.__dict__.items():
            if type(val) == Variable or type(val) == VariableGroup:
                val.save()

    # def print_all(self):




if __name__ == '__main__':
    import csdl_alpha as csdl

    @dataclass
    class MyVG(VariableGroup):
        a : Union[Variable, int, float]
        b : Variable

        def define(self):
            self.declare_parameters('a', shape=(1,), variablize=True)
            self.declare_parameters('b', type=Variable, shape=(1,))

    recorder = csdl.Recorder()
    recorder.start()

    my_vg = MyVG(a=1, b=csdl.Variable(shape=(1,), value=1))

    b = my_vg.b