def test_graph_building():
    import csdl_alpha as csdl
    from csdl_alpha.src.graph.variable import Variable

    recorder = csdl.build_new_recorder()
    recorder.start()
    a = Variable((1,), name='a')
    csdl.enter_namespace('test')
    assert recorder.active_namespace.name == 'test'
    b = Variable((1,), name='b')
    csdl.exit_namespace()
    assert recorder.active_namespace.name == 'root'
    recorder.stop()

    assert len(recorder.graph.nodes) == 2

    print(recorder.graph.nodes)


test_graph_building()




# test
