
from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.node import Node
from csdl_alpha.src.operations.loops.new_loop.new_loop import NewLoop
from csdl_alpha.src.operations.implicit_operations.implicit_operation import ImplicitOperation 

def is_var(node:Node)->bool:
    """
    Check if a node is a variable.
    """
    if isinstance(node, Variable):
        return True
    return False

def is_op(node:Node)->bool:
    """
    Check if a node is an operation.
    """
    if isinstance(node, Operation):
        return True
    return False

def is_loop(node:Node)->bool:
    """
    Check if a node is a loop.
    """
    if isinstance(node, NewLoop):
        return True
    return False

def is_implicit(node:Node)->bool:
    """
    Check if a node is an implicit operation.
    """
    if isinstance(node, ImplicitOperation):
        return True
    return False