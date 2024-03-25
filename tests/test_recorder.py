
def test_subgraphs():
    from csdl_alpha.src.graph.variable import Variable
    import csdl_alpha as csdl

    recorder = csdl.build_new_recorder()
    recorder.start()
    a = Variable((1,), name='a')
    assert len(recorder.active_graph.node_table) == 1
    csdl.enter_subgraph()
    assert len(recorder.active_graph.node_table) == 0
    b = Variable((1,), name='b')
    assert len(recorder.active_graph.node_table) == 1
    csdl.exit_subgraph()
    assert len(recorder.active_graph.node_table) == 1
    c = Variable((1,), name='c')
    assert len(recorder.active_graph.node_table) == 2
    recorder.stop()


def test_namespacing():
    import csdl_alpha as csdl
    from csdl_alpha.src.graph.variable import Variable

    recorder = csdl.build_new_recorder()
    recorder.start()
    a = Variable((1,), name='a')
    csdl.enter_namespace('test1')
    assert recorder.active_namespace.name == 'test1'
    b = Variable((1,), name='b')
    csdl.enter_namespace('test2')
    assert recorder.active_namespace.name == 'test2'
    csdl.exit_namespace()
    assert recorder.active_namespace.name == 'test1'
    csdl.exit_namespace()
    assert recorder.active_namespace.name == 'root'
    recorder.stop()

    assert a.namespace.name == 'root'
    assert b.namespace.name == 'test1'
    assert b.namespace.prepend == 'root.test1'
    assert len(recorder.active_graph.node_table) == 2


test_namespacing()
test_subgraphs()



# test
