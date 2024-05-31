'''
def test_visualization(tmp_path):
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()
    import os
    os.chdir(tmp_path)

    import csdl_alpha as csdl

    recorder = csdl.Recorder()
    recorder.start()

    x = csdl.Variable(name = "x", value = 1)
    y = csdl.Variable(name = "y", value = 2)
    z = x*y
    z.add_name("z")

    viz_test_objects = [
        csdl.visualize_graph,
        recorder.visualize_graph,
        recorder.active_graph.visualize,
    ]

    for i, VIZ_OBJECT_func in enumerate(viz_test_objects):
        VIZ_OBJECT_func()
        VIZ_OBJECT_func(trim_loops = True)
        VIZ_OBJECT_func(trim_loops = True, format = 'svg')
        
        if i <= 1:
            VIZ_OBJECT_func(visualize_style = 'hierarchical')
            VIZ_OBJECT_func(visualize_style = 'hierarchical', filename = 'test')

    recorder.stop()
'''
