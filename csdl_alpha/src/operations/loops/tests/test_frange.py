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
        recorder.visualize_graph('test_frange')

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
                a = a.set((i, j), i+j)

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
                a = a.set((i, j), b[i,j] * i+j)
        
        a_np = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(a, a_np)
            ]
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
        self.prep()
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

        assert a.value == np.array([6])
        assert b.value == np.array([22])
        assert c.value == np.array([38])

    def test_stack(self):
        self.prep()
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

        loop_vars = loop.op.loop_vars
        b_stack = loop_vars[1][2]
        assert np.all(b_stack.value == np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]))
        assert b_stack == loop.op.get_stacked(b)

if __name__ == '__main__':
    test = Testfrange()
    test.test_simple_loop()
    test.test_simple_double_loop()
    test.test_range_inputs()
    test.test_setitem()
    test.test_setget()
    # test.test_loop_var_history()
    # test.test_compute_iteration()
    test.test_custom_vals()
    test.test_multi_vals()
    test.test_stack()

# class TestVRange(csdl_tests.CSDLTest):
#     def test_simple_loop(self):
#         self.prep()
#         import csdl_alpha as csdl
#         from csdl_alpha.api import vrange
#         import numpy as np

#         a = csdl.Variable(value=2, name='a')
#         b = b0 = csdl.Variable(value=3, name='b')
#         for i in vrange(0, 10):
#             b2 = a + b
#             c = a*2
        
#         x = a+b2+b0+c

#         self.run_tests(
#             compare_values=[
#                 csdl_tests.TestingPair(a, np.array([2])),
#                 csdl_tests.TestingPair(b, np.array([3])),
#                 csdl_tests.TestingPair(b0, np.array([3])),
#                 csdl_tests.TestingPair(b2, np.array([5])),
#                 csdl_tests.TestingPair(c, np.array([2*2])),
#                 csdl_tests.TestingPair(x, np.array([2+5+3+4]))
#             ]
#         )

#     def test_setitem(self):
#         self.prep()
#         import csdl_alpha as csdl
#         from csdl_alpha.api import vrange
#         import numpy as np

#         a = csdl.Variable(shape=(3,3), value=12*np.ones((3,3)))
        
#         for i in vrange(0, 3):
#             for j in vrange(0, 3):
#                 b = a.set((i, j), i+j)

#         self.run_tests(
#             compare_values=[
#                 csdl_tests.TestingPair(b, np.array([[12, 12, 12], [12, 12, 12], [12, 12, 4]]))
#             ]
#         )



#     def test_range_inputs(self):
#         self.prep()
#         import csdl_alpha as csdl
#         from csdl_alpha.api import vrange

#         with pytest.raises(ValueError):
#             vrange(10, 0)

#         v_range = vrange(vals=[1, 2, 3, 4, 5])
#         assert v_range.vals == [1, 2, 3, 4, 5]



# Tests with feedback - turned off right now
# class TestVRange(csdl_tests.CSDLTest):
#     def test_simple_loop(self):
#         self.prep()
#         import csdl_alpha as csdl
#         from csdl_alpha.api import vrange
#         import numpy as np

#         a = csdl.Variable(value=2, name='a')
#         b = b0 = csdl.Variable(value=3, name='b')
#         for i in vrange(0, 10):
#             b = a + b
#             c = a*2
        
#         x = a+b+b0+c

#         self.run_tests(
#             compare_values=[
#                 csdl_tests.TestingPair(a, np.array([2])),
#                 csdl_tests.TestingPair(b, np.array([3+2*10])),
#                 csdl_tests.TestingPair(b0, np.array([3])),
#                 csdl_tests.TestingPair(c, np.array([2*2])),
#                 csdl_tests.TestingPair(x, np.array([2+(3+2*10)+3+2*2]))
#             ]
#         )

#     def test_range_inputs(self):
#         self.prep()
#         import csdl_alpha as csdl
#         from csdl_alpha.api import vrange

#         with pytest.raises(ValueError):
#             vrange(10, 0)

#         v_range = vrange(vals=[1, 2, 3, 4, 5])
#         assert v_range.vals == [1, 2, 3, 4, 5]
