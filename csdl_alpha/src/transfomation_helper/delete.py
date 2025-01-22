from csdl_alpha.src.transfomation_helper.checks import is_op
from csdl_alpha.src.graph.node import Node

def delete_nodes(nodes:list, force_delete:bool = False):
    if not isinstance(nodes, list):
        if not isinstance(nodes, Node):
            raise TypeError("nodes must be a Node object or a list of Node objects.")
        nodes = [nodes]

    import csdl_alpha as csdl
    recorder = csdl.get_current_recorder()
    for node in nodes:
        recorder.delete_node(node, force_delete=force_delete)

def delete_unvalued_descendants(recorder = None):
    """
    Deletes all nodes in the graph that cannot be evaluated due to missing values. For example:

    x = csdl.Variable(name = 'x')
    y = csdl.Variable(name = 'y', value = 1.0)
    z1 = x + y
    z2 = x + 1.0

    ct = csdl.transformation_helper
    ct.delete_unvalued_descendants() # z2 and the last addition will be deleted as x is unvalued
    """

    if recorder is None:
        import csdl_alpha as csdl
        recorder = csdl.get_current_recorder()

    # Loop through all input leaf nodes
    # if a node is not valued, delete it and all of its descendants
    delete_nodes_set = set()
    for node in recorder.get_root_graph().node_table:
        if recorder.get_root_graph().in_degree(node) == 0:
            if node.value is None:
                delete_nodes_set.add(node)
                descendants = recorder.get_root_graph().descendants(node)
                delete_nodes_set.update(descendants)
    delete_nodes(list(delete_nodes_set), force_delete=True)