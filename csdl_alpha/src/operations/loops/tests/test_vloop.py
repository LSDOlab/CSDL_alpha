from csdl_alpha.src.operations.loops.vloop import BLoop
from csdl_alpha.utils.hard_reload import hard_reload
import csdl_alpha.utils.testing_utils as csdl_tests
import pytest
import numpy as np
def build_bloop_1():
    import csdl_alpha as csdl
    recorder = csdl.get_current_recorder()

    shape = (2,2)
    size = 4
    batch_size = 3

    # unbatched inputs:
    av = np.arange(size).reshape(shape)
    bv = -np.arange(size).reshape(shape)
    cv = np.ones(shape)/3.0
    a_dim = csdl.Variable(value=av, name='a')
    b_dim = csdl.Variable(value=bv, name='b')
    c_dim = csdl.Variable(value=cv, name='c')

    # batched inputs:
    xv = np.arange(size*batch_size).reshape((batch_size, *shape))
    yv = np.arange(size*batch_size).reshape((batch_size, *shape))+0.1
    x = csdl.Variable(value=xv, name='x')
    y = csdl.Variable(value=yv, name='y')

    # batching body:
    recorder._enter_subgraph(add_missing_variables=True)
    x_dim = csdl.Variable(value=np.ones(shape), name='x_dim')
    y_dim = csdl.Variable(value=np.ones(shape), name='y_dim')
    u = a_dim+c_dim+x_dim**1.0
    w = b_dim+c_dim+y_dim
    w_o = w.flatten()[0]
    w_o.add_name('w_o')
    u.add_name('u')
    body_graph = recorder.active_graph
    recorder._exit_subgraph()

    # batched outputs:
    batched_u = csdl.Variable(value=np.ones((batch_size, *shape)), name='u_batched')
    batched_w = csdl.Variable(value=np.ones((batch_size,1)), name='w_batched')

    bw_sum = 0
    for i in range(batch_size):
        bw_val = (bv+cv+yv[i]).flatten()[0]
        bw_sum += bw_val

    out_analytical = (av+cv+xv)*bw_sum
    batch_u_analytical = av+cv+xv
    batch_w_analytical = (bv+cv+yv)[:,0,0].reshape(-1,1)

    outs = {
        batched_u: batch_u_analytical,
        batched_w: batch_w_analytical
    }

    return body_graph, [(x, x_dim), (y, y_dim)], [a_dim, b_dim, c_dim], [(batched_u, u), (batched_w, w_o)], outs


def build_bloop_2():
    import csdl_alpha as csdl
    recorder = csdl.get_current_recorder()

    shape = (2,)
    size = 2
    batch_size = 3

    # unbatched inputs:
    av = np.arange(size).reshape(shape)
    bv = np.array([0.3])
    a_dim = csdl.Variable(value=av, name='a')
    b_dim = csdl.Variable(value=bv, name='b')

    # batched inputs:
    xv = np.arange(size*batch_size).reshape((batch_size, *shape))
    yv = np.arange(size*batch_size).reshape((batch_size, *shape))+0.1
    x = csdl.Variable(value=xv, name='x')
    y = csdl.Variable(value=yv, name='y')

    # batching body:
    recorder._enter_subgraph(add_missing_variables=True)
    x_dim = csdl.Variable(value=np.ones(shape), name='x_dim')
    y_dim = csdl.Variable(value=np.ones(shape), name='y_dim')
    u = csdl.norm(a_dim*x_dim)
    w_o = b_dim**2.0+y_dim**2.0*x_dim
    w_o.add_name('w_o')
    u.add_name('u')
    body_graph = recorder.active_graph
    recorder._exit_subgraph()

    # batched outputs:
    batched_u = csdl.Variable(value=np.ones((batch_size,1)), name='u_batched')
    batched_w = csdl.Variable(value=np.ones((batch_size,*shape)), name='w_batched')

    batch_u_analytical = np.linalg.norm(av*xv, axis = 1).reshape(-1,1)
    batch_w_analytical = bv**2.0+yv**2.0*xv

    print(batch_w_analytical)
    outs = {
        batched_u: batch_u_analytical,
        batched_w: batch_w_analytical
    }

    return body_graph, [(x, x_dim), (y, y_dim)], [a_dim, b_dim], [(batched_u, u), (batched_w, w_o)], outs


def test_bloop():
    import numpy as np
    import csdl_alpha as csdl
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    shape = (2,2)
    size = 4
    batch_size = 3

    # unbatched inputs:
    av = np.arange(size).reshape(shape)
    bv = -np.arange(size).reshape(shape)
    cv = np.ones(shape)/3.0
    a_dim = csdl.Variable(value=av, name='a')
    b_dim = csdl.Variable(value=bv, name='b')
    c_dim = csdl.Variable(value=cv, name='c')

    # batched inputs:
    xv = np.arange(size*batch_size).reshape((batch_size, *shape))
    yv = np.arange(size*batch_size).reshape((batch_size, *shape))+0.1
    x = csdl.Variable(value=xv, name='x')
    y = csdl.Variable(value=yv, name='y')

    # batching body:
    recorder._enter_subgraph(add_missing_variables=True)
    x_dim = csdl.Variable(value=np.ones(shape), name='x_dim')
    y_dim = csdl.Variable(value=np.ones(shape), name='y_dim')
    u = a_dim+c_dim+x_dim
    w = b_dim+c_dim+y_dim
    w_o = w.flatten()[0]
    w_o.add_name('w_o')
    u.add_name('u')
    body_graph = recorder.active_graph
    recorder._exit_subgraph()

    # batched outputs:
    batched_u = csdl.Variable(value=np.ones((batch_size, *shape)), name='u_batched')
    batched_w = csdl.Variable(value=np.ones((batch_size,1)), name='w_batched')
    output = batched_u*csdl.sum(batched_w)
    
    bloop_op = BLoop(
        body=body_graph,
        batched_inputs=[(x, x_dim), (y, y_dim)],
        unbatched_inputs=[a_dim, b_dim, c_dim],
        batched_outputs=[(batched_u, u), (batched_w, w_o)],
    )
    bloop_op.finalize_and_return_outputs()
    # recorder.visualize_graph(visualize_style='hierarchical')

    recorder.execute()

    # Check results:
    bw_sum = 0
    for i in range(batch_size):
        bw_val = (bv+cv+yv[i]).flatten()[0]
        np.testing.assert_almost_equal(batched_u.value[i], av+cv+xv[i])
        np.testing.assert_almost_equal(batched_w.value[i], bw_val)
        bw_sum += bw_val

    out_analytical = (av+cv+xv)*bw_sum
    batch_u_analytical = av+cv+xv
    batch_w_analytical = (bv+cv+yv)[:,0,0].reshape(-1,1)

    np.testing.assert_almost_equal(output.value, out_analytical)
    np.testing.assert_almost_equal(batched_u.value, batch_u_analytical)
    np.testing.assert_almost_equal(batched_w.value, batch_w_analytical)

    jax_interface = csdl.jax.create_jax_interface(
        [a_dim, b_dim, c_dim, x, y],
        [output, batched_u, batched_w],
        recorder.active_graph,
    )

    outputs = jax_interface({a_dim:av, b_dim:bv, c_dim:cv, x:xv, y:yv})
    np.testing.assert_almost_equal(outputs[output], out_analytical)
    np.testing.assert_almost_equal(outputs[batched_u], batch_u_analytical)
    np.testing.assert_almost_equal(outputs[batched_w], batch_w_analytical)

class Testvrange(csdl_tests.CSDLTest):
    def test_simple(self):
        self.prep(inline = False)
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np
        recorder = csdl.get_current_recorder()

        body, ins, s_ins, b_outs, outs = build_bloop_1()
        bloop = BLoop(
            body=body,
            batched_inputs=ins,
            unbatched_inputs=s_ins,
            batched_outputs=b_outs,
        )
        batched_u, batched_w = bloop.finalize_and_return_outputs()
        output = batched_u*csdl.sum(batched_w)
        output.name = 'output'
        outs[output] = outs[batched_u]*np.sum(outs[batched_w])
        
        # csdl.derivative(output, ins[0][0])
        # exit()

        compare_values = []
        for out, val in outs.items():
            compare_values.append(csdl_tests.TestingPair(out, val, tag = out.name))

        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')
        # self.run_tests(compare_values)
        self.run_tests(compare_values, verify_derivatives=True)

    def test_simple2(self):
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        body, ins, s_ins, b_outs, outs = build_bloop_2()
        bloop = BLoop(
            body=body,
            batched_inputs=ins,
            unbatched_inputs=s_ins,
            batched_outputs=b_outs,
        )
        batched_u, batched_w = bloop.finalize_and_return_outputs()
        output = batched_u*csdl.sum(batched_w)
        output.name = 'output'
        outs[output] = outs[batched_u]*np.sum(outs[batched_w])
        
        compare_values = []
        for out, val in outs.items():
            compare_values.append(csdl_tests.TestingPair(out, val, tag = out.name))
        self.run_tests(compare_values, verify_derivatives=True)

    # def test_nested(self):
    #     self.prep()
    #     import csdl_alpha as csdl
    #     from csdl_alpha.api import frange
    #     import numpy as np

    #     body, ins, s_ins, b_outs, outs = build_bloop_2()
    #     bloop = BLoop(
    #         body=body,
    #         batched_inputs=ins,
    #         unbatched_inputs=s_ins,
    #         batched_outputs=b_outs,
    #     )
    #     batched_u, batched_w = bloop.finalize_and_return_outputs()
    #     output = batched_u*csdl.sum(batched_w)
    #     output.name = 'output'
    #     outs[output] = outs[batched_u]*np.sum(outs[batched_w])
        
    #     compare_values = []
    #     for out, val in outs.items():
    #         compare_values.append(csdl_tests.TestingPair(out, val, tag = out.name))
    #     self.run_tests(compare_values)

if __name__ == '__main__':
    t = Testvrange()
    t.overwrite_backend = 'jax'
    t.test_simple()
    t.test_simple2()
