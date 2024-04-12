# VariableGroup

The `VariableGroup` class is a container for `Variable` objects. It is used to group variables together, which can be useful for organizing your model. The `VariableGroup` class can be used as inputs and outputs to functions, and contains methods to assign metadata to variables in the group.

## Creating a VariableGroup

There are two main ways to create and use VariableGroups. The first is to instantiate a 'VariableGroup' object and add variables to it:

```python
import csdl_alpha as csdl

recorder = csdl.Recorder()
recorder.start()

vg = csdl.VariableGroup()
x = csdl.Variable(value=0)
y = csdl.Variable(value=0)
vg.x = x
vg.y = y

vg.add_tag('input')
```

The second way is to create a subclass of VariableGroup and define the variables in the subclass. This is useful for adding checks to the variables in the group, such as type and shape. These checks are performed when the variables are added to the group, and can be used to ensure that the variables are valid. The dataclass decorator must be used for subclasses of VariableGroup, and you can pass arguments into the decorator to further define your VariableGroup.

```python
import csdl_alpha as csdl
from dataclasses import dataclass
from csdl_alpha.utils.typing import VariableLike

@dataclass
class CustomVariableGroup(csdl.VariableGroup):
    a : VariableLike
    b : Variable
    def define_checks(self):
        self.add_check('a', shape=(1,), variablize=True)
        self.add_check('b', type=csdl.Variable, shape=(1,))

a = csdl.Variable(value=0)
b = csdl.Variable(value=0)

vg = CustomVariableGroup(a=a, b=b)

vg.add_tag('input')
```