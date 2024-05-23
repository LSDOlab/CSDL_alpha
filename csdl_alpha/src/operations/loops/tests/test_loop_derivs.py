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

        with csdl.namespace('deriv1'):
            deriv = csdl.derivative(x, a)
        with csdl.namespace('deriv2'):
            deriv2 = csdl.derivative(deriv, a)
        with csdl.namespace('deriv3'):
            deriv3 = csdl.derivative(deriv2, a)
        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')

        # print('COMPUTING SECOND DERIVATIVES')
        #     deriv23 = csdl.derivative(deriv, loop.op.get_stacked(b))
        #     # deriv23 = csdl.derivative(loop.op.get_stacked(b), a)
        # print(deriv2.value)
        # print(deriv23.value)
        # from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
        # verify_derivatives_inline(ofs=[csdl.derivative(x, a)], wrts=[a], step_size=1e-6)
        # exit()
            
        # manual derivative check:
        # --- forward --- :
        # x     = a +       b      +  c
        # x     = a + (5a^3+a^4/2) + (-a)
        # --- derivs --- :
        # dx/da    = 15a^2 + 2a^3
        # dx2/da2  = 30*a + 6a^2
        # dx3/da3  = 30 + 12a
        
        deriv_value = (15*a.value**2.0+2*a.value**3)
        assert abs(deriv.value[0,0] - deriv_value) < 1e-9
        deriv_value2 = (30*a.value+6*a.value**2)
        assert abs(deriv2.value[0,0] - deriv_value2) < 1e-9
        deriv_value3 = (30+12*a.value)
        assert abs(deriv3.value[0,0] - deriv_value3) < 1e-9

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(x, x.value),
                csdl_tests.TestingPair(deriv, deriv_value.reshape(1,1)),
                csdl_tests.TestingPair(deriv3, deriv_value3.reshape(1,1)),
            ],
            verify_derivatives=True
        )

    def test_simple_second_deriv(self):
        self.prep()
        import csdl_alpha as csdl
        import numpy as np

        b0 = csdl.Variable(value = 1.2, name='b0')
        b = b0*1.0
        b.add_name(f'b_ext_in')

        loop = csdl.frange(0, 3)
        for i in loop:
            b.add_name(f'b*')
            b = b**2.0
        b.add_name('b_updated')
        x = -b
        x.add_name('x')

        with csdl.namespace('deriv1'):
            deriv = csdl.derivative(x, b0)
        with csdl.namespace('deriv2'):
            deriv2 = csdl.derivative(deriv, b0)
        with csdl.namespace('deriv3'):
            deriv3 = csdl.derivative(deriv2, b0)
        with csdl.namespace('deriv4'):
            deriv4 = csdl.derivative(deriv3, b0)
        # manual derivative check:
        # --- forward --- :
        # x     = -b0
        # x     = -((b0^2)^2)^2 = -b0^8
        # --- derivs --- :
        # dx/db0 = -8*b0^7
        # dx2/db02 = -56*b0^6
        # dx3/db03 = -336*b0^5
        # dx4/db04 = -1680*b0^4
        
        deriv_value = -8.0*(b0.value**7.0)
        deriv_value2 = -56.0*(b0.value**6.0)
        deriv_value3 = -336.0*(b0.value**5.0)
        deriv_value4 = -1680.0*(b0.value**4.0)

        print(deriv.value, deriv_value)
        print(deriv2.value, deriv_value2)
        print(deriv3.value, deriv_value3)
        print(deriv4.value, deriv_value4)
        assert abs(deriv.value[0,0] - deriv_value) < 1e-9
        assert abs(deriv2.value[0,0] - deriv_value2) < 1e-9
        assert abs(deriv3.value[0,0] - deriv_value3) < 1e-9
        assert abs(deriv4.value[0,0] - deriv_value4) < 1e-9

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(x, x.value),
                csdl_tests.TestingPair(deriv, deriv.value),
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
        deriv2 = csdl.derivative(deriv, b0)

        # manual derivative check
        # x     =     b      +        c
        # x     =  (12b0^8)  +  (12b0^8 + b0)
        # x     =  24b0^8 + b0
        # dx/db = 192b0^7 + 1
        # dx2/db2 = 1344b0^6

        analyt_deriv = 192*b0.value**7.0+1
        assert abs(deriv.value[0,0] - (analyt_deriv)) < 1e-9
        analyt_deriv2 = 1344*b0.value**6.0
        assert abs(deriv2.value[0,0] - (analyt_deriv2)) < 1e-9

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(x, x.value),
                csdl_tests.TestingPair(deriv, analyt_deriv.reshape(1,1)),
                csdl_tests.TestingPair(deriv2, analyt_deriv2.reshape(1,1)),
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

        a.add_name('a_updated')
        x = csdl.sum(a)
        x.add_name('x')
        deriv = csdl.derivative([a, x], a_0)
        da_da0 = deriv[a]
        dx_da0 = deriv[x]

        recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')
        # loop.op.get_subgraph().visualize('temp')
        a_0.set_value(np.arange(6).reshape(3,2)*0.01+0.2)
        recorder.execute()

        a_val = np.arange(6).reshape(3,2)*0.01+0.2
        b_val = a_val
        for i in range(2):
            a_val = a_val + b_val*0.5

        # TODO: manual derivative check
        # x     = a +       b      +  c
        # x     = a + (5a^3+a^4/2) + (-a)
        # dx/da = 15a^2 + 2a^3
        # assert abs(deriv.value[0,0] - (15*a.value**2.0+2*a.value**3)) < 1e-9

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(x, np.sum(a_val).flatten()),
                csdl_tests.TestingPair(da_da0, da_da0.value),
                csdl_tests.TestingPair(dx_da0, dx_da0.value),
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

        deriv = csdl.derivative(x, a_0)

        # TODO: manual derivative check
        # assert abs(deriv.value[0,0] - (15*a.value**2.0+2*a.value**3)) < 1e-9

        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(x, x.value),
                csdl_tests.TestingPair(c, c.value),
                csdl_tests.TestingPair(deriv, deriv.value),
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

        deriv = csdl.derivative([x, b], a_0)
        deriv_x = deriv[x]
        deriv_b = deriv[b]
        # TODO: manual derivative check
        # assert abs(deriv.value[0,0] - (15*a.value**2.0+2*a.value**3)) < 1e-9
        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')
        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(x, x.value),
                csdl_tests.TestingPair(c, c.value),
                csdl_tests.TestingPair(deriv_x, deriv_x.value),
                csdl_tests.TestingPair(deriv_b, deriv_b.value),
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
        c.add_name('c_updated')
        x = csdl.sum(b+a+c)
        for i in csdl.frange(2):
            x = x + a[i,0]
            b = x/c
        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')
        # deriv = csdl.derivative(c, [a_0, d])
        # exit()
        deriv = csdl.derivative([x, b, c], [a_0, d])
        dx_da0 = csdl.norm(deriv[x, a_0])
        dx_d = csdl.norm(deriv[x, d])
        db_da0 = csdl.norm(deriv[b, a_0])
        db_d = csdl.norm(deriv[b, d])
        dc_da0 = csdl.norm(deriv[c, a_0])
        dc_d = csdl.norm(deriv[c, d])
        deriv_sums = dx_da0+dx_d+db_da0+db_d+dc_da0+dc_d
        deriv_sums.add_name('deriv_sums')

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

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(x, x_val.reshape(x.shape)),
                csdl_tests.TestingPair(c, c_val.reshape(c.shape)),
                csdl_tests.TestingPair(a, a_val.reshape(a.shape)),
                csdl_tests.TestingPair(d, np.array(d_val).reshape(d.shape)),
                csdl_tests.TestingPair(deriv_sums, deriv_sums.value, decimal=4),
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

        deriv = csdl.derivative([x, b, c], [a_0, d])
        dx_da0 = csdl.norm(deriv[x, a_0])
        dx_d = csdl.norm(deriv[x, d])
        db_da0 = csdl.norm(deriv[b, a_0])
        db_d = csdl.norm(deriv[b, d])
        dc_da0 = csdl.norm(deriv[c, a_0])
        dc_d = csdl.norm(deriv[c, d])
        deriv_sums = dx_da0+dx_d+db_da0+db_d+dc_da0+dc_d
        deriv_sums.add_name('deriv_sums')

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
                csdl_tests.TestingPair(deriv_sums, deriv_sums.value, decimal=4),
            ],
            verify_derivatives=True
        )

if __name__ == '__main__':
    t = TestFrangeDeriv()
    # t.test_simple_loop()
    # t.test_simple_second_deriv()
    # t.test_simple_loop2()
    # t.test_simple_loop_feedback()
    # t.test_simple_loop_feedback_indexing()
    # t.test_simple_loop_feedback_indexing2()
    # t.test_nested()
    t.test_nested_double_indexing()
