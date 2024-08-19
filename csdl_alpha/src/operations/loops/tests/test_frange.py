import csdl_alpha.utils.testing_utils as csdl_tests
import pytest

class Testfrange(csdl_tests.CSDLTest):
    def test_simple_loop(self):
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        a = csdl.Variable(value=2, name='a')
        b = b0 = csdl.Variable(value=3, name='b')
        for i in frange(0, 10):
            b = a + b
            c = a*2
        x = a+b

        a_np = np.array([2])
        b_np = b0_np = np.array([3])
        for i in range(0, 10):
            b_np = a_np + b_np
            c_np = a_np*2
        x_np = a_np+b_np

        recorder = csdl.get_current_recorder()

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(a, a_np),
                csdl_tests.TestingPair(b, b_np),
                csdl_tests.TestingPair(b0, b0_np),
                csdl_tests.TestingPair(c, c_np),
                csdl_tests.TestingPair(x, x_np)
            ]
        )

    def test_simple_double_loop(self):
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        a = csdl.Variable(value=2, name='a')
        b = b0 = csdl.Variable(value=3, name='b')
        for i in frange(0, 10):
            for j in frange(0, 10):
                b = a + b
                c = a*2
        x = a+b

        a_np = np.array([2])
        b_np = b0_np = np.array([3])
        for i in range(0, 10):
            for j in range(0, 10):
                b_np = a_np + b_np
                c_np = a_np*2
        x_np = a_np+b_np

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(a, a_np),
                csdl_tests.TestingPair(b, b_np),
                csdl_tests.TestingPair(b0, b0_np),
                csdl_tests.TestingPair(c, c_np),
                csdl_tests.TestingPair(x, x_np)
            ]
        )

    def test_range_inputs(self):
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.api import frange

        with pytest.raises(ValueError):
            frange(10, 0)

        f_range = frange(vals=[1, 2, 3, 4, 5])
        assert f_range.vals[0] == [1, 2, 3, 4, 5]

    def test_setitem(self):
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        a = csdl.Variable(shape=(3,3), value=12*np.ones((3,3)))
        
        for i in frange(0, 3):
            for j in frange(0, 3):
                a = a.set(csdl.slice[(i, j)], i+j)

        a_np = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(a, a_np)
            ]
        )
    
    def test_setget(self):
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        a = csdl.Variable(shape=(3,3), value=12*np.ones((3,3)))
        b = csdl.Variable(shape=(3,3), value=np.ones((3,3)))
        
        for i in frange(0, 3):
            for j in frange(0, 3):
                a = a.set(csdl.slice[(i, j)], b[i,j] * i+j)
        
        a_np = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(a, a_np)
            ]
        )

    def test_loop_constant(self):
        # added after bug found by Nick
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        a = csdl.Variable(value=1)
        b = 0
        x = csdl.Variable(value=2)

        for i in csdl.frange(10):
            b = b+a*x

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(b, np.array([20]))
            ],
            verify_derivatives=True
        )


    def test_no_feedback(self):
        # added after bug found by Nick
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        a = csdl.Variable(value=1)
        x = csdl.Variable(value=2)

        for i in csdl.frange(10):
            b = a*x
        c = b*1.0
        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(b, np.array([2])),
                csdl_tests.TestingPair(c, np.array([2]))
            ],
            verify_derivatives=True
        )



    # def test_loop_var_history(self):
    #     self.prep()
    #     import csdl_alpha as csdl
    #     from csdl_alpha.api import frange
    #     import numpy as np

    #     a = csdl.Variable(value=2, name='a')
    #     b = csdl.Variable(value=3, name='b')
    #     loop_i = frange(0, 10)
    #     for i in loop_i:
    #         b = a + b
    #     b_history = list(loop_i.op.loop_var_history.values())[0]

    #     a_np = np.array([2.])
    #     b_np = np.array([3.])
    #     b_history_np = []
    #     for i in range(0, 10):
    #         b_history_np.append(b_np)
    #         b_np = a_np + b_np
            
    #     for b, b_np in zip(b_history, b_history_np):
    #         assert np.allclose(b, b_np)


    # def test_compute_iteration(self):
    #     self.prep()
    #     import csdl_alpha as csdl
    #     from csdl_alpha.api import frange
    #     import numpy as np

    #     a = csdl.Variable(value=2, name='a')
    #     b = csdl.Variable(value=3, name='b')
    #     loop_i = frange(0, 10)
    #     for i in loop_i:
    #         b = a + b
    #     b_history = list(loop_i.op.loop_var_history.values())[0]

    #     b_history_recomputed = [np.array([3.])]
    #     for i in range(9):
    #         loop_i.op.compute_iteration(i)
    #         b_history_recomputed.append(b.value)
            
    #     for b, b_recomp in zip(b_history, b_history_recomputed):
    #         assert np.allclose(b, b_recomp)

    def test_custom_vals(self):
        self.prep(always_build_inline = True)
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        a = csdl.Variable(value=0, name='a')
        for i in frange(vals=[0,1,2,3]):
            a = a + i

        assert a.value == np.array([6])

    def test_multi_vals(self):
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        a = csdl.Variable(value=0, name='a')
        b = csdl.Variable(value=0, name='b')
        c = csdl.Variable(value=0, name='c')
        for i, j in frange(vals=([0,1,2,3], [4,5,6,7])):
            a = a + i
            b = b + j
            c = c + i*j

        compare_values = []
        compare_values += [csdl_tests.TestingPair(a, np.array([6]))]
        compare_values += [csdl_tests.TestingPair(b, np.array([22]))]
        compare_values += [csdl_tests.TestingPair(c, np.array([38]))]
        self.run_tests(compare_values=compare_values, verify_derivatives=True)

    def test_multi_vals_delay_inline(self):
        self.prep(always_build_inline = True)
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        a = csdl.Variable(value=0, name='a')
        b = csdl.Variable(value=0, name='b')
        c = csdl.Variable(value=0, name='c')

        # loop = frange(vals=([0,1,2,3], [4,5,6,7]), inline_lazy_stack=True)
        loop = frange(vals=([0,1,2,3], [4,5,6,7]))
        for i, j in loop:
            a = a + i
            b = b + j
            c = c + i*j

        # assert loop.op.inline_lazy_stack == True
        # assert loop.op.outputs[-1].value is None
        # loop.op.get_stacked(a)
        # assert loop.op.inline_lazy_stack == False

        compare_values = []
        compare_values += [csdl_tests.TestingPair(a, np.array([6]))]
        compare_values += [csdl_tests.TestingPair(b, np.array([22]))]
        compare_values += [csdl_tests.TestingPair(c, np.array([38]))]
        self.run_tests(compare_values=compare_values, verify_derivatives=True)

    def test_multi_vals_delay_inline_nest(self):
        self.prep(always_build_inline = True)
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        a = csdl.Variable(value=0, name='a')
        b = csdl.Variable(value=0, name='b')
        c = csdl.Variable(value=0, name='c')
        e = csdl.Variable(value=0.2, name='c')

        # loop = frange(vals=([0,1,2,3], [4,5,6,7]), inline_lazy_stack=True)
        loop = frange(vals=([0,1,2,3], [4,5,6,7]))
        for i, j in loop:
            a = a + i*e*5.0
            for k in frange(0, 1):
                a = a**1.0
                d = a+1.0
            b = b + j
            c = c + i*j

        # assert loop.op.inline_lazy_stack == True
        # assert loop.op.outputs[-1].value is None
        # loop.op.get_stacked(a)
        # assert loop.op.inline_lazy_stack == False

        deriv = csdl.derivative(c, [a])

        compare_values = []
        compare_values += [csdl_tests.TestingPair(a, np.array([6]))]
        compare_values += [csdl_tests.TestingPair(b, np.array([22]))]
        compare_values += [csdl_tests.TestingPair(c, np.array([38]))]
        compare_values += [csdl_tests.TestingPair(d, np.array([7]))]
        self.run_tests(compare_values=compare_values, verify_derivatives=True)

    def test_multi_vals_delay_inline_nest2(self):
        self.prep(always_build_inline = True)
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        a = csdl.Variable(value=0, name='a')
        b = csdl.Variable(value=0, name='b')
        c = csdl.Variable(value=0, name='c')
        e = csdl.Variable(value=0.2, name='c')

        # loop = frange(vals=([0,1,2,3], [4,5,6,7]), inline_lazy_stack=True)
        loop = frange(vals=([0,1,2,3], [4,5,6,7]))
        for i, j in loop:
            a = a + i*e*5.0
            # for k in frange(0, 1, inline_lazy_stack=True):
            for k in frange(0, 1):
                a = a**1.0
                d = a+1.0
            b = b + j
            c = c + i*j

        # assert loop.op.inline_lazy_stack == True
        # assert loop.op.outputs[-1].value is None
        # loop.op.get_stacked(a)
        # assert loop.op.inline_lazy_stack == False

        deriv = csdl.derivative(c, [a])

        compare_values = []
        compare_values += [csdl_tests.TestingPair(a, np.array([6]))]
        compare_values += [csdl_tests.TestingPair(b, np.array([22]))]
        compare_values += [csdl_tests.TestingPair(c, np.array([38]))]
        compare_values += [csdl_tests.TestingPair(d, np.array([7]))]
        self.run_tests(compare_values=compare_values, verify_derivatives=True)

    def test_stack(self):
        self.prep(always_build_inline = True)
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        a = csdl.Variable(value=1, name='a')
        b = csdl.Variable(value=1, name='b')
        loop = frange(0,10)
        
        for i in loop:
            b = a + b
            c = a*2
        x = a+b

        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph('stacked', visualize_style='hierarchical')
        lb = loop.op.loop_builder
        b_stack = lb.stacked[lb.feedbacks._output_to_feedback[b].internal_input]
        assert np.all(b_stack.value == np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]))

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(b, np.array([11])),
                csdl_tests.TestingPair(b_stack, np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]))
            ],
            verify_derivatives=True
        )

    def test_stack_multi(self):
        self.prep(always_build_inline = True)
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        a = csdl.Variable(value=1, name='a')
        b = csdl.Variable(value=1, name='b')
        c = csdl.Variable(value=4, name='c')
        loop = frange(0,10)
        
        for i in loop:
            b = a + b
            c = b+c
        x = a+b

        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph('stacked', visualize_style='hierarchical')

        lb = loop.op.loop_builder
        b_stack = lb.stacked[lb.feedbacks._output_to_feedback[b].internal_input]
        c_stack = lb.stacked[lb.feedbacks._output_to_feedback[c].internal_input]

        real_b = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        real_c = [4]
        for i in range(9):
            real_c.append(real_b[i+1][0]+real_c[i])

        assert np.all(b_stack.value == real_b)
        assert np.all(c_stack.value == np.array(real_c).reshape(10,1))

    def test_stack_multidim(self):
        self.prep(always_build_inline = True)
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        a_val = np.arange(6).reshape(2,3)-1
        b_val = np.arange(6).reshape(2,3)
        c_val = np.ones(3)*3.0
        c_val[0] = 2.0

        a = csdl.Variable(value=a_val, name='a')
        b = csdl.Variable(value=b_val, name='b')
        c = csdl.Variable(value=c_val, name='c')
        loop = frange(0,4)
        
        for i in loop:
            b = a + b
            c = b[1]+c
            b = b + csdl.expand(c, out_shape=(2,3), action='i->ji')
        x = a+b

        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph('stacked', visualize_style='hierarchical')

        lb = loop.op.loop_builder
        b_stack = lb.stacked[lb.feedbacks._output_to_feedback[b].internal_input]
        c_stack = lb.stacked[lb.feedbacks._output_to_feedback[c].internal_input]

        real_b = np.zeros((4,2,3))
        real_c = np.zeros((4,3))
        for i in range(4):
            real_b[i] = b_val
            real_c[i] = c_val
            b_val = a_val + b_val
            c_val = b_val[1]+c_val
            b_val = b_val + c_val

        assert np.all(b_stack.value == real_b)
        assert np.all(c_stack.value == real_c)

    def test_feedback(self):
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.api import frange
        import numpy as np

        a_val = np.arange(6).reshape(2,3)-1
        b_val = np.arange(6).reshape(2,3)
        c_val = np.ones((2,3))*3.0
        c_val[0] = 2.0

        a = csdl.Variable(value=a_val, name='a')
        b = csdl.Variable(value=b_val, name='b')
        c = csdl.Variable(value=c_val, name='c')

        loop = frange(0,4)
        for i in loop:
            b.add_name('b_in')
            c.add_name('c_in')
            c = c + c + b
            b = a + b

        c.add_name('c_updated')
        b.add_name('b_updated')

        # loop_vars = loop.op.loop_vars
        # for i,loop_var in enumerate(loop_vars):
        #     print(f'loop var {i}')
        #     print(f'--{loop_var[0].name}')
        #     print(f'--{loop_var[1].name}')
        #     print(f'--{loop_var[2].name}')

        lb = loop.op.loop_builder
        assert len(lb.feedbacks._int_input_to_feedback) == 2


        for i in range(4):
            c_val = c_val + c_val + b_val
            b_val = a_val + b_val

        compare_values = []
        compare_values += [csdl_tests.TestingPair(c, c_val)]
        compare_values += [csdl_tests.TestingPair(b, b_val)]

        self.run_tests(compare_values=compare_values)

    # TODO: add pure accrue function for loops
    # def test_pure_accrue(self):
    #     self.prep()
    #     import csdl_alpha as csdl
    #     from csdl_alpha.api import frange
    #     import numpy as np

    #     a_val = 3.14
    #     b_val = 0.0

    #     a0 = csdl.Variable(value=a_val, name='a')
    #     b0 = csdl.Variable(value=b_val, name='b')

    #     a = a0
    #     b = b0

    #     b.add_name('b_init')
    #     loop = frange(0,3)
    #     for i in loop:
    #         a = a**2.0

    #         a.add_name('a_updated') 
    #         b.add_name('b_body_in')
    #         b = b + csdl.sin(a)
    #         b.add_name('b_updated')

    #     with csdl.namespace('derivatives'):
    #         deriv = csdl.derivative(csdl.cos(b), [a0])
    #         print(deriv[a0].value)

    #         # print(csdl.derivative(deriv[a0], loop.op.get_stacked(b)).value)
    #     # recorder = csdl.get_current_recorder()
    #     # recorder.visualize_graph(visualize_style='hierarchical')
    #     exit('TODO: add pure accrue function for loops')
    #     compare_values = []
    #     compare_values += [csdl_tests.TestingPair(b, b_val)]

    #     self.run_tests(compare_values=compare_values, verify_derivatives=True)

if __name__ == '__main__':
    test = Testfrange()
    test.overwrite_backend = 'jax'
    # test.test_simple_loop()
    # test.test_simple_double_loop()
    # test.test_range_inputs()
    # test.test_setitem()
    # test.test_setget()
    # test.test_custom_vals()
    # test.test_multi_vals()
    # test.test_stack()
    # test.test_stack_multi()
    # test.test_stack_multidim()
    # test.test_feedback()
    # test.test_pure_accrue()
    test.test_implicit_in_loop()