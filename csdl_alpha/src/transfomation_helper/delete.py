     

def delete_nodes(nodes:list):
    if not isinstance(nodes, list):
        nodes = [nodes]

    import csdl_alpha as csdl
    recorder = csdl.get_current_recorder()
    for node in nodes:
        recorder.delete_node(node)