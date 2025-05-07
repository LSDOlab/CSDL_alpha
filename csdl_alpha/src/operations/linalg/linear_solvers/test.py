# import csdl_alpha.utils.testing_utils as csdl_tests

# class TestLinear(csdl_tests.CSDLTest):
    
    # def test_functionality(self,):
    #     self.prep()

    #     import csdl_alpha as csdl
    #     import numpy as np
    #     x_val = 3.0
    #     y_val = 2.0
    #     x = csdl.Variable(name = 'x', value = x_val)
    #     y = csdl.Variable(name = 'y', value = y_val)
        
    #     compare_values = []
    #     # add scalar variables
    #     s1 = csdl.add(x,y)
    #     t1 = np.array([x_val + y_val])
    #     compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]


    #     self.run_tests(compare_values = compare_values, verify_derivatives=True)

    # def test_docstring(self):
        # self.docstest(add)

# if __name__ == '__main__':
#     test = TestLinear()
    # test.overwrite_backend = 'jax'
    # test.test_functionality()
    # test.test_docstring()