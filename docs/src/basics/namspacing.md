# Namespacing

CSDL supports namespacing to organize your model. While your code can be organized into functions and classes, namespacing organizes your model after is is compiled into a graph.

## Namespacing with Namespace context manager

The `Namespace` class provides a context manager to create a namespace. The `Namespace` class is used to create a namespace, and the `with` statement is used to enter and exit the namespace. The `Namespace` class takes a name as an argument, which is used to name the namespace. The `Namespace` class can be used to create nested namespaces, which can be useful for organizing your model.

Namespaces are applied to any variables or operations created within the namespace. Additionally, any variable that is given a name within a namespace has that namespace applied to it, regardless of where the variable was created. Variables can have multiple names with different namespaces, but the variable itself has a single namespace.
<!-- does that make sense? -->

```python
import csdl_alpha as csdl

recorder = csdl.Recorder()
recorder.start()

with csdl.Namespace('namespace_1'):
    x = csdl.Variable(value=0, name='x')
    y = csdl.Variable(value=0 , name='y')
    with csdl.Namespace('namespace_2'):
        x.add_name('a')
        z = x + y
        z.add_name('z')

recorder.stop()

print(x.names)
print(z.names)
```

```sh
$ python3 ex_namespace.py
['namespace_1.x', 'namespace_1.namespace_2.a']
['namespace_1.namespace_2.z']

```

