import csdl_alpha as csdl
import numpy as np
import csdl_alpha.utils.testing_utils as csdl_tests

class TestDeriv(csdl_tests.CSDLTest):
    def test_deriv(self):
        """
        Test derivatives with sequences of operations
        """
        self.prep()

        # test basics
        x0 = csdl.Variable(value=np.arange(6).reshape(3,2)+1.0)
        x1 = csdl.Variable(value=2.0)

        y = csdl.norm(x0[[0,1], :]) + csdl.sin(x1)
        # print(np.sin(2.0))
        # print(np.cos(2.0))
        # print(np.linalg.norm(np.sin(2.0)))

        dy_dx = csdl.derivative(y, [x0, x1])
        dy_dx0 = dy_dx[x0]
        dy_dx1 = dy_dx[x1]

        compare_values = []
        y_real = np.linalg.norm(np.arange(4).reshape(2,2)+1.0) + np.sin(2.0)
        compare_values += [csdl_tests.TestingPair(y, y_real.reshape(1,))]
        compare_values += [csdl_tests.TestingPair(dy_dx1, np.cos(2.0).reshape(1,1))]

        self.run_tests(compare_values=compare_values, verify_derivatives=True)

    def test_deriv_2(self):
        """
        Test second derivatives with composed operations etc
        """
        self.prep()

        # test basics
        x0 = csdl.Variable(value=np.arange(6).reshape(3,2)+1.0)
        x1 = csdl.Variable(value=2.0)
        x0.add_name('x0')
        x1.add_name('x1')

        y = csdl.norm(x0[[0,1], :]) - csdl.sin(x1)
        # y = x0+x1
        y.add_name('y')
        with csdl.namespace('deriv1'):
            dy_dx1 = csdl.derivative(y, [x1, x0])[x1]
            dy_dx1.add_name('dy_dx1')

        with csdl.namespace('deriv2'):
            d2y_dx12 = csdl.derivative(dy_dx1, [x1])[x1]
            d2y_dx12.add_name('d2y_dx12')

        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph()
        # exit()
        compare_values = []
        y_real = np.linalg.norm(np.arange(4).reshape(2,2)+1.0) - np.sin(2.0)
        compare_values += [csdl_tests.TestingPair(y, y_real.reshape(1,))]
        compare_values += [csdl_tests.TestingPair(d2y_dx12, d2y_dx12.value)]

        self.run_tests(compare_values=compare_values, verify_derivatives=True)

    def test_deriv_3(self):
        """
        Test more sequences with different operations
        """
        self.prep()

        x0 = csdl.Variable(value=np.arange(6).reshape(3,2)+1.0)
        x1 = csdl.Variable(value=2.0)

        y = x0.set(csdl.slice[1:2], csdl.sum(x0))
        y1 = y-csdl.exp(x1)**(x1*x1)
        y2 = csdl.tensordot(y1, x0, axes=([1],[1]),)
        with csdl.namespace('deriv1'):
            dy_dx1 = csdl.derivative(y2, [x1, x0])[x1]
            dy_dx1.add_name('dy_dx1')

        y3 = csdl.maximum(csdl.absolute(dy_dx1/100000))
        compare_values = []
        # compare_values += [csdl_tests.TestingPair(dy_dx1, dy_dx1.value)]
        compare_values += [csdl_tests.TestingPair(y3, y3.value)]
        self.run_tests(compare_values=compare_values, verify_derivatives=True)

    def test_deriv_composed(self):
        """
        Test single derivatives with composed operations
        """
        self.prep()

        x0 = csdl.Variable(name = 'x0', value=1.0)
        x1 = csdl.Variable(name = 'x1', value=2.0)

        y2 = x1-x0
        y2.add_name('y2')
        with csdl.namespace('deriv1'):
            dy_dx1 = csdl.derivative(y2, [x1, x0])[x1]
            dy_dx1.add_name('dy_dx1')

        with csdl.namespace('deriv2'):
            dy_dx2 = csdl.derivative(dy_dx1, [x1])[x1]
            dy_dx2.add_name('dy2_dx2')

        compare_values = []
        compare_values += [csdl_tests.TestingPair(dy_dx2, dy_dx2.value)]
        compare_values += [csdl_tests.TestingPair(dy_dx1, np.ones((1,1)))]
        self.run_tests(compare_values=compare_values, verify_derivatives=True)

    def test_deriv_composed2(self):
        """
        Test single derivatives with composed operations
        """
        self.prep()

        x0 = csdl.Variable(name = 'x0', value=np.array([1.0, 2.0]))
        x1 = csdl.Variable(name = 'x1', value=np.array([3.0, -2.0]))

        # y2 = x1-x1
        y2 = csdl.tensordot(x1, x1)
        y2.add_name('y2')
        with csdl.namespace('deriv1'):
            dy_dx1 = csdl.derivative(y2, [x1, x0])[x1]
            dy_dx1.add_name('dy_dx1')

        with csdl.namespace('deriv2'):
            dy_dx2 = csdl.derivative(csdl.sum(dy_dx1), [x1])[x1]
            dy_dx2.add_name('dy2_dx2')

        compare_values = []
        compare_values += [csdl_tests.TestingPair(dy_dx2, dy_dx2.value)]
        compare_values += [csdl_tests.TestingPair(dy_dx1, dy_dx1.value)]
        self.run_tests(compare_values=compare_values, verify_derivatives=True)

    def test_deriv_composed3(self):
        """
        Test single derivatives with composed operations
        """
        self.prep(inline=False)

        x0 = csdl.Variable(name = 'x0', value=np.array([1.0, 2.0]))
        x1 = csdl.Variable(name = 'x1', value=np.array([3.0, -2.0]))

        # y2 = x1-x1
        # y2 = csdl.tensordot(x1, x1)
        y2 = csdl.einsum(x1, x0, action = 'i,j->ij')
        y2.add_name('y2')
        with csdl.namespace('deriv1'):
            dy_dx1 = csdl.derivative(csdl.sum(y2), [x1])[x1]
            dy_dx1.add_name('dy_dx1')

        with csdl.namespace('deriv2'):
            dy_dx2 = csdl.derivative(csdl.sum(dy_dx1), [x1])[x1]
            dy_dx2.add_name('dy2_dx2')

        with csdl.namespace('deriv3'):
            dy_dx3 = csdl.derivative(csdl.sum(dy_dx2), [x1])[x1]
            dy_dx3.add_name('dy2_dx3')

        # recorder = csdl.get_current_recorder()
        # # recorder.stop()
        # # recorder.visualize_graph()
        # # exit()
        
        # # print(dy_dx2.value)
        # x0.value = np.array([2.0, 2.0])
        # # recorder.execute()
        # # print(dy_dx2.value)
        # # recorder.execute()
        # recorder.active_graph.execute_inline(debug=True)
        # print('========================================')
        # recorder.active_graph.execute_inline(debug=True)

        # exit()

        # compare_values = []
        # compare_values += [csdl_tests.TestingPair(dy_dx2, dy_dx2.value)]
        # compare_values += [csdl_tests.TestingPair(dy_dx1, dy_dx1.value)]
        self.run_tests(compare_values=[], verify_derivatives=True)

    def test_deriv_composed5(self):
        """
        Test single derivatives with composed operations
        """
        self.prep(inline=True)

        x1 = csdl.Variable(name = 'x1', value=np.array([3.0, -2.0]))

        y2 = x1*x1
        # y2 = csdl.tensordot(x1, x1)
        # y2 = csdl.einsum(x1, x0, action = 'i,j->ij')
        y2.add_name('y2')
        with csdl.namespace('deriv1'):
            dy_dx1 = csdl.derivative((y2), [x1])[x1]
            dy_dx1.add_name('dy_dx1')

        with csdl.namespace('deriv2'):
            sum_dy_dx1 = (dy_dx1)
            sum_dy_dx1.add_name('sum_dy_dx1')
            dy_dx2 = csdl.derivative(sum_dy_dx1, [x1])[x1]
            dy_dx2.add_name('dy2_dx2')

        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical')
        # stack_out_jac_y2_wrt_x1 = recorder._find_variables_by_name('deriv1.stack_out_jac_y2_wrt_x1')[0]
        # seed_y2 = recorder._find_variables_by_name('deriv1.seed_y2')[0]
        # from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
        # verify_derivatives_inline([dy_dx2], [seed_y2], step_size=1e-6, raise_on_error=False)
        
        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(visualize_style='hierarchical', filename='2')
        # recorder.stop()

        # exit()
        
        # # print(dy_dx2.value)
        # x0.value = np.array([2.0, 2.0])
        # # recorder.execute()
        # # print(dy_dx2.value)
        # # recorder.execute()
        # recorder.active_graph.execute_inline(debug=True)
        # print('========================================')
        # recorder.active_graph.execute_inline(debug=True)

        # exit()

        compare_values = []
        compare_values += [csdl_tests.TestingPair(dy_dx1, dy_dx1.value)]
        compare_values += [csdl_tests.TestingPair(dy_dx2, dy_dx2.value)]
        self.run_tests(compare_values=compare_values, verify_derivatives=True)

    def test_deriv_composed4(self):
        """
        Taking 3rd derivatives and running it for a third time seemed to have issues...
        """
        self.prep(inline=True)

        x0 = csdl.Variable(name = 'x0', value=np.array([1.0, 2.0]))
        x1 = csdl.Variable(name = 'x1', value=np.array([3.0, 1.0]))

        # Composed Operations:
        # y2 = x1-x1
        y2 = csdl.tensordot(x1, x1)
        # y2 = csdl.einsum(x1, x0, action = 'i,j->ij')
        # y2 = csdl.outer(x1, x1)
        # y2 = csdl.minimum(x1, x1)
        # y2 = csdl.exp(x1)
        # y2 = csdl.sum(x1, x1)

        y2.add_name('y2')
        with csdl.namespace('deriv1'):
            dy2_dx1 = csdl.derivative(y2[0], x1)
            dy2_dx1.add_name('dy_dx1')

        with csdl.namespace('deriv2'):
            dy2_dx2 = csdl.derivative(dy2_dx1[0], x1)
            dy2_dx2.add_name('dy2_dx2')

        with csdl.namespace('deriv3'):
            dy3_dx2 = csdl.derivative(dy2_dx2[0], x1)
            dy3_dx2.add_name('dy3_dx3')
            print(dy3_dx2.shape)

        recorder = csdl.get_current_recorder()
        recorder.stop()
        x0.value = np.array([2.0, 2.0])
        recorder.active_graph.execute_inline(debug=False)
        recorder.active_graph.execute_inline(debug=False)


if __name__ == '__main__':
    t = TestDeriv()
    # t.test_deriv()
    # t.test_deriv_2()
    # t.test_deriv_3()
    # t.test_deriv_composed()
    # t.test_deriv_composed2()
    # t.test_deriv_composed3()
    # t.test_deriv_composed4()
    # t.test_deriv_composed5()






