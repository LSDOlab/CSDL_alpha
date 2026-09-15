
from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.node import Node
from csdl_alpha.src.operations.loops.new_loop.new_loop import NewLoop
from csdl_alpha.src.operations.implicit_operations.implicit_operation import ImplicitOperation 

from typing import Any

def is_node(node:Any)->bool:
    """
    Check if a node is a node.
    """
    if isinstance(node, Node):
        return True
    return False

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

def predecessors(node:Node, graph = None)->list[Node]:
    """
    Get the predecessors of a node.
    """
    if graph is None:
        import csdl_alpha as csdl
        graph = csdl.get_current_recorder().get_root_graph()
    
    if node not in graph.node_table:
        raise ValueError(f'Node {node.info()} not in the graph')
    
    preds = list(graph.predecessors(node))
    return preds