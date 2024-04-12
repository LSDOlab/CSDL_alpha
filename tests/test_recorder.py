import pytest

def test_subgraphs():
    from csdl_alpha.src.graph.variable import Variable
    import csdl_alpha as csdl

    recorder = csdl.Recorder()
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


def test_with_namespacing():
    import csdl_alpha as csdl

    recorder = csdl.Recorder()
    recorder.start()
    a = csdl.Variable((1,), name='a')
    with csdl.Namespace('test1'):
        b = csdl.Variable((1,), name='b')
        with csdl.Namespace('test2'):
            c = csdl.Variable((1,), name='c')
        d = csdl.Variable((1,), name='d')
    e = csdl.Variable((1,), name='e')
    recorder.stop()
    assert a.namespace.name is None
    assert b.namespace.name == 'test1'
    assert c.namespace.name == 'test2'
    assert c.namespace.prepend == 'test1.test2'
    assert d.namespace.name == 'test1'
    assert e.namespace.name is None

def test_namespacing():
    import csdl_alpha as csdl
    from csdl_alpha.src.graph.variable import Variable

    recorder = csdl.Recorder()
    recorder.start()
    with pytest.raises(Exception):
        csdl.exit_namespace()
    with pytest.raises(TypeError):
        csdl.enter_namespace(1)
    a = Variable((1,), name='a')
    csdl.enter_namespace('test1')
    assert recorder.active_namespace.name == 'test1'
    b = Variable((1,), name='b')
    csdl.enter_namespace('test2')
    assert recorder.active_namespace.name == 'test2'
    c = Variable((1,), name='c')
    csdl.exit_namespace()
    assert recorder.active_namespace.name == 'test1'
    csdl.exit_namespace()
    assert recorder.active_namespace.name is None
    recorder.stop()

    assert a.namespace.name is None
    assert b.namespace.name == 'test1'
    assert b.namespace.prepend == 'test1'
    assert c.namespace.prepend == 'test1.test2'
    assert len(recorder.active_graph.node_table) == 3

def test_duplicate_namespace_error():
    import csdl_alpha as csdl
    from csdl_alpha.src.graph.variable import Variable

    recorder = csdl.Recorder()
    recorder.start()
    csdl.enter_namespace('test1')
    csdl.exit_namespace()
    with pytest.raises(Exception) as e_info:
        csdl.enter_namespace('test1')

def test_auto_hierarchy():
    import csdl_alpha as csdl
    from csdl_alpha.src.graph.variable import Variable

    recorder = csdl.Recorder(auto_hierarchy=True)
    recorder.start()
    a = Variable((1,), name='a')
    assert a.hierarchy == 0
    csdl.enter_namespace('test1')
    b = Variable((1,), name='b')
    assert b.hierarchy == 1
    csdl.enter_namespace('test2')
    c = Variable((1,), name='c')
    assert c.hierarchy == 2
    csdl.exit_namespace()
    d = Variable((1,), name='d')
    assert d.hierarchy == 1
    csdl.exit_namespace()
    e = Variable((1,), name='e')
    assert e.hierarchy == 0
    recorder.stop()
