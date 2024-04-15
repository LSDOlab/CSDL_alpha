import csdl_alpha.utils.test_utils as csdl_tests
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
        assert f_range.vals == [1, 2, 3, 4, 5]

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
