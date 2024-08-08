import csdl_alpha.utils.testing_utils as csdl_tests
import pytest

class Testfrange(csdl_tests.CSDLTest):
    def test_implicit_in_loop(self):
        import csdl_alpha as csdl

        # added after bug in MBD
        solvers = [
            ({'stack_all':True}, csdl.nonlinear_solvers.Newton),
            ({'stack_all':False}, csdl.nonlinear_solvers.Newton),
            ({'stack_all':True}, csdl.nonlinear_solvers.GaussSeidel),
            ({'stack_all':False}, csdl.nonlinear_solvers.GaussSeidel),
        ]

        for loop_kwargs, nlsolver in solvers:
            self.prep(always_build_inline=True)
            from csdl_alpha.api import frange
            import numpy as np

            out_val = np.array([1.58890495])
            out_val_prev = np.array([1.60106013])
            num = 10
            a_val = 1.5+np.arange(num)
            a = csdl.Variable(name = 'a', value = 1.5+np.arange(10)/20)
            b = csdl.Variable(name = 'b', value = 0.5)
            c = csdl.Variable(name = 'c', value = -1.0)

            x_initial = csdl.Variable(shape=(1,), name='x_initial', value=0.34)
            loop = csdl.frange(10, **loop_kwargs)
            i = 0

            for t in loop:
                x_initial_1 = x_initial*1
                x_initial_1.add_name('x_initial_1')

                x = csdl.ImplicitVariable(shape=(1,), name='x', value=0.34)

                ax2 = a[t]*x**3.0
                y = (-ax2 - c)*b - x + 5.0
                y.name = 'residual_x'

                # apply coupling:
                # ONE SOLVER COUPLING:
                solver = nlsolver(nlsolver.__name__)
                x_initial.add_name('x_initial_'+str(i))
                i += 1
                if isinstance(solver, csdl.nonlinear_solvers.Newton):
                    solver.add_state(
                        x, 
                        y, 
                        initial_value=x_initial_1,
                    )
                else:
                    solver.add_state(
                        x, 
                        y, 
                        initial_value=x_initial_1,
                        state_update = x+0.1*y,
                    )

                solver.run()

                x_initial = x
            
            deriv = csdl.derivative(x, a)
            print('previous iteration:', x_initial_1.value)
            print('last iteration:',x.value)
            self.run_tests(
                compare_values=[
                    csdl_tests.TestingPair(x, out_val, decimal = 7),
                    csdl_tests.TestingPair(x_initial_1, out_val_prev, decimal = 7),
                    csdl_tests.TestingPair(deriv, deriv.value, decimal = 7),
                ],
                verify_derivatives = True,
            )

    def test_simple_loop_feedback_indexing(self):
        self.prep(always_build_inline = True, debug = True)
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        # a = a*b[i,0] + b[i+1,1]
        # a = a.set([i, 0], a[i,0]+1.0)

        a_val = np.arange(6).reshape(3,2)*0.01+0.1
        a_0 = csdl.Variable(value=a_val, name='a_0')
        a = a_0*1.0
        b = a
        b.add_name(f'b_in')
        loop_frange = csdl.frange(2, stack_all=True)
        for i in loop_frange:
            l = i+1
            a = a*b[i,0] + b[l,1]
            b = a+b
            a = a.set(csdl.slice[i, 0], a[i,0])
            a = a+a
            for ii in csdl.frange(3, stack_all=True):
                a = a+ii
            c = b*a
            c.add_name(f'c')
        print(loop_frange.op.loop_builder)
        a.add_name('a_updated')
        x = csdl.sum(b+a+c)
        x.add_name('x')

        deriv = csdl.derivative(x, a_0)
        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')
        # deriv = csdl.derivative(deriv, a_0)
        # exit()
        # TODO: manual derivative check
        # assert abs(deriv.value[0,0] - (15*a.value**2.0+2*a.value**3)) < 1e-9


        # check real:
        a_0np = a_val*1.0
        anp = a_0np*1.0
        bnp = anp
        for i in range(2):
            anp = anp*bnp[i,0] + bnp[i+1,1]
            bnp = anp+bnp
            anp[i,0] = anp[i,0]
            anp= anp+anp
            for ii in range(3):
                anp = anp+ii
            cnp = bnp*anp
        xnp  = np.sum(bnp +anp +cnp)
        xnp = np.array(xnp).reshape(x.shape)

        # print(xnp,x.value)
        assert np.isclose(xnp, x.value).all()
        assert np.isclose(cnp, c.value).all()

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(x, x.value),
                csdl_tests.TestingPair(c, c.value),
                csdl_tests.TestingPair(deriv, deriv.value),
            ],
            verify_derivatives=True
        )

if __name__ == '__main__':
    t = Testfrange()
    t.overwrite_backend = 'jax'
    # t.test_implicit_in_loop()
    t.test_simple_loop_feedback_indexing()