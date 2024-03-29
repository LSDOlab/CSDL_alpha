
# def set_properties(
#         linear: bool = False,
#         elementwise: bool = False,
#         diagonal_jacobian: bool = False,
#         convex: bool = False,
#         elementary: bool = True,
#         supports_sparse: bool = False,
#         contains_subgraph: bool = False
#     ):

#     properties = {
#         'linear': linear,
#         'elementwise': elementwise,
#         'diagonal_jacobian': diagonal_jacobian,
#         'convex': convex,
#         'elementary': elementary,
#         'supports_sparse': supports_sparse,
#         'contains_subgraph': contains_subgraph
#     }
        
#     def decorator(cls):
#         cls.properties = properties
#         return cls
#     return decorator




class Parent(object):
    properties = {
        'linear': False,
        'elementwise': False,
        'diagonal_jacobian': False,
        'convex': False,
        'elementary': True,
        'supports_sparse': False,
    }

def set_properties(**kwargs):
    for property, value in kwargs.items():
        if not isinstance(property, str):
            raise ValueError("Property names must be strings")
        if property not in Parent.properties:
            raise ValueError(f"Property {property} not recognized. Must be one of {Parent.properties.keys()}")
        if not isinstance(value, bool):
            raise ValueError("Property values must be boolean")
        
    def decorator(cls):
        properties = cls.properties.copy()
        for property, value in kwargs.items():
            properties[property] = value
        cls.properties = properties
        return cls
    return decorator

@set_properties(linear=True, elementwise=True)
class Child(Parent):
    pass

@set_properties(diagonal_jacobian=True)
class GrandChild(Child):
    pass

parent = Parent()
child = Child()
grandchild = GrandChild()
grandchild2 = GrandChild()

grandchild.properties['linear'] = False
print(parent.properties)
print(child.properties)
print(grandchild.properties)
print(grandchild2.properties) # kind of annoything
