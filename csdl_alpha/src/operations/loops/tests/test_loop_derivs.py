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

    def test_simple_loop_feedback(self):
        self.prep()
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
        loop = csdl.frange(0, 2)
        for i in loop:
            c = csdl.Variable(value=0.5, name='c')
            a = a+b*c

        recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')
        # loop.op.get_subgraph().visualize('temp')
        a_0.set_value(np.arange(6).reshape(3,2)*0.01+0.2)
        recorder.execute()

        a_val = np.arange(6).reshape(3,2)*0.01+0.2
        b_val = a_val
        for i in range(2):
            a_val = a_val + b_val*0.5

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
                csdl_tests.TestingPair(x, np.sum(a_val).flatten()),
            ],
            verify_derivatives=True
        )

    def test_simple_loop_feedback_indexing(self):
        self.prep()
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
        for i in csdl.frange(2):
            a = a*b[i,0] + b[i+1,1]
            b = a+b
            a = a.set(csdl.slice[i, 0], a[i,0])
            a = a+a
            c = b*a
        a.add_name('a_updated')
        x = csdl.sum(b+a+c)
        x.add_name('x')

        # TODO: manual derivative check
        # assert abs(deriv.value[0,0] - (15*a.value**2.0+2*a.value**3)) < 1e-9

        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(x, x.value),
                csdl_tests.TestingPair(c, c.value),
            ],
            verify_derivatives=True
        )

    def test_simple_loop_feedback_indexing2(self):
        self.prep()
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
        for i in csdl.frange(2):
            a = a*b[i,0] + b[i+1,1]
            b = a+b
            a = a.set(csdl.slice[i, 0], a[i,0])
            a = a+a
            c = b*a
        a.add_name('a_updated')
        x = csdl.sum(b+a+c)

        for i in csdl.frange(2):
            x = x + a[i,0]
            b = x/c
        # TODO: manual derivative check
        # assert abs(deriv.value[0,0] - (15*a.value**2.0+2*a.value**3)) < 1e-9
        recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')
        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(x, x.value),
                csdl_tests.TestingPair(c, c.value),
            ],
            verify_derivatives=True
        )

    def test_nested(self):
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np
        a_val = np.arange(6).reshape(3,2)*0.01+0.1
        a_0 = csdl.Variable(value=a_val, name='a_0')
        d = csdl.Variable(value=0.2, name='d')
        a = a_0*1.0
        b = a
        b.add_name(f'b_in')
        for i in csdl.frange(2):
            a = a*b[i,0] + b[i+1,1]
            b = a+b
            a = a.set(csdl.slice[i, 0], a[i,0])
            for j in csdl.frange(2):
                a = a.set(csdl.slice[i, j], a[i,j]*b[i+1,j])
                for k in csdl.frange(3):
                    d = 1.2*d/(k+1)
            a = a+a+d
            c = b*a
        a.add_name('a_updated')
        d.add_name('d_updated')
        x = csdl.sum(b+a+c)
        for i in csdl.frange(2):
            x = x + a[i,0]
            b = x/c

        recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')
        a_0.set_value(np.arange(6).reshape(3,2)*0.01+0.15)
        recorder.execute()

        a_val = np.arange(6).reshape(3,2)*0.01+0.15
        a_val = a_val*1.0
        b_val = a_val
        d_val = 0.2
        for i in range(2):
            a_val = a_val*b_val[i,0] + b_val[i+1,1]
            b_val = a_val+b_val
            a_val[i, 0] = a_val[i,0]
            for j in range(2):
                a_val[i, j] = a_val[i,j]*b_val[i+1,j]
                for k in range(3):
                    d_val = 1.2*d_val/(k+1)
            a_val = a_val+a_val+d_val
            c_val = b_val*a_val
        x_val = np.sum(b_val+a_val+c_val)
        for i in range(2):
            x_val = x_val + a_val[i,0]
            b_val = x_val/c_val

        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')
        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(x, x_val.reshape(x.shape)),
                csdl_tests.TestingPair(c, c_val.reshape(c.shape)),
                csdl_tests.TestingPair(a, a_val.reshape(a.shape)),
                csdl_tests.TestingPair(d, np.array(d_val).reshape(d.shape)),
            ],
            verify_derivatives=True
        )

    def test_nested_double_indexing(self):
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np
        a_val = np.arange(8).reshape(4,2)*0.01+0.1
        a_0 = csdl.Variable(value=a_val, name='a_0')
        d_0 = csdl.Variable(value=0.2, name='d')
        d = d_0*1.0
        a = a_0*1.0
        b = a
        b.add_name(f'b_in')
        for i,j in csdl.frange(vals = ([0,2,1], [1,0,0])):
            a = a*b[i,0] + b[i+1,1]
            b = a+b
            a = a.set(csdl.slice[i, 0], a[i,0])
            for k, l in csdl.frange(vals = ([0,1], [1,0])):
                a = a.set(csdl.slice[k, j], a[l,j]*b[i+1,j])
                for k in csdl.frange(3):
                    d = 1.2*d/(k+1)
            a = a+a+d
            c = b*a
        x = csdl.sum(b+a*c)
        a.add_name('a_updated')
        d.add_name('d_updated')

        recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')
        a_0.set_value(np.arange(8).reshape(4,2)*0.01+0.015)
        recorder.execute()

        a_val = np.arange(8).reshape(4,2)*0.01+0.015
        a_val = a_val*1.0
        b_val = a_val
        d_val = 0.2
        for i,j in zip([0,2,1], [1,0,0]):
            a_val = a_val*b_val[i,0] + b_val[i+1,1]
            b_val = a_val+b_val
            a_val[i, 0] = a_val[i,0]
            for k,l in zip([0,1], [1,0]):
                a_val[k, j] = a_val[l,j]*b_val[i+1,j]
                for k in range(3):
                    d_val = 1.2*d_val/(k+1)
            a_val = a_val+a_val+d_val
            c_val = b_val*a_val
        x_val = np.sum(b_val+a_val*c_val)
        
        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')
        
        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(x, x_val.reshape(x.shape)),
                csdl_tests.TestingPair(c, c_val.reshape(c.shape)),
                csdl_tests.TestingPair(a, a_val.reshape(a.shape)),
                csdl_tests.TestingPair(d, np.array(d_val).reshape(d.shape)),
            ],
            verify_derivatives=True
        )

if __name__ == '__main__':
    t = TestFrangeDeriv()
    # t.test_simple_loop()
    # t.test_simple_loop2()
    # t.test_simple_loop_feedback()
    # t.test_simple_loop_feedback_indexing()
    # t.test_simple_loop_feedback_indexing2()
    # t.test_nested()
    # t.test_nested_double_indexing()
