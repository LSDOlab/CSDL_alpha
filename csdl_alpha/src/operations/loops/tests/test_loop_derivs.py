import csdl_alpha.utils.testing_utils as csdl_tests
import pytest

class TestFrangeDeriv(csdl_tests.CSDLTest):
    def test_simple_loop(self):
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        a = csdl.Variable(value=1.2, name='a')
        b0 = csdl.Variable(value=5, name='b')
        b = b0*1.0+a*0.5
        b.add_name(f'b_ext_in')

        loop = csdl.frange(0, 3)
        for i in loop:
            b.add_name(f'b*')
            b = a*b
            c = -a
        b.add_name('b_updated')
        c.add_name('c_updated')
        x = a+b+c

        deriv = csdl.derivative(x, a)

        # manual derivative check
        # x     = a +       b      +  c
        # x     = a + (5a^3+a^4/2) + (-a)
        # dx/da = 15a^2 + 2a^3
        assert abs(deriv.value[0,0] - (15*a.value**2.0+2*a.value**3)) < 1e-9

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(x, x.value),
            ],
            verify_derivatives=True
        )

    def test_simple_loop2(self):
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        b0 = csdl.Variable(value=1.1, name='b')
        b = b0*1.0
        b.add_name(f'b_ext_in')

        loop = csdl.frange(0, 3)
        for i in loop:
            b.add_name(f'b*')
            b = (i+1)*(b*b)
            c = b+b0
        b.add_name('b_updated')
        c.add_name('c_updated')
        x = b+c
        print(b.value)

        deriv = csdl.derivative(x, b0)

        # manual derivative check
        # x     =     b      +        c
        # x     =  (12b0^8)  +  (12b0^8 + b0)
        # x     =  24b0^8 + b0
        # dx/db = 192b0^7 + 1
        assert abs(deriv.value[0,0] - (192*b0.value**7.0+1)) < 1e-9

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(x, x.value),
            ],
            verify_derivatives=True
        )

    def test_simple_loop_indexing_feedback(self):
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        # a = a*b[i,0] + b[i+1,1]
        # a = a.set([i, 0], a[i,0]+1.0)

        a_val = np.arange(6).reshape(3,2)*0.01+0.1
        a_val = 0.1
        a_0 = csdl.Variable(value=a_val, name='a_0')
        a = a_0*1.0
        b = a
        b.add_name(f'b_in')
        loop = csdl.frange(0, 2)
        for i in loop:
            c = csdl.Variable(value=0.5, name='c')
            a = a+b*c

            # a = a+b*0.5

        print(a.value)
        print(b.value)
        print()
        recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')
        # loop.op.get_subgraph().visualize('temp')
        a_0.set_value(0.2)
        recorder.execute()
        print(a.value)
        print(b.value)
        print()

        a_val = 0.2
        b_val = a_val
        for i in range(2):
            a_val = a_val + b_val*0.5
        print(a_val)
        print(b_val)
        # exit()
        loop_vars = loop.op.loop_vars
        for i,loop_var in enumerate(loop_vars):
            print(f'loop var {i}')
            print(f'--{loop_var[0]}')
            print(f'--{loop_var[1]}')
            print(f'--{loop_var[2]}')

        # exit()

        # b.add_name('b_updated')
        a.add_name('a_updated')
        x = csdl.sum(a)
        x.add_name('x')

        # TODO: manual derivative check
        # x     = a +       b      +  c
        # x     = a + (5a^3+a^4/2) + (-a)
        # dx/da = 15a^2 + 2a^3
        # assert abs(deriv.value[0,0] - (15*a.value**2.0+2*a.value**3)) < 1e-9

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(x, x.value),
            ],
            verify_derivatives=True
        )

if __name__ == '__main__':
    t = TestFrangeDeriv()
    t.test_simple_loop()
    # t.test_simple_loop2()
    # t.test_simple_loop_indexing_feedback()