from csdl_alpha.src.graph.operation import Operation, set_properties
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests
import csdl_alpha as csdl

import numpy as np

class Maximum(Operation):
    '''
    Maximum entries in the input tensor along the specified axes.
    '''
    def __init__(self, x, axes=None, out_shape=None, rho=20.):
        super().__init__(x)
        self.name  = 'maximum'
        out_shapes = (out_shape,)
        self.set_dense_outputs(out_shapes)
        self.axes  = axes
        self.rho   = rho
        in_shape = x.shape

        if axes is not None:
            axes = self.axes  = tuple(np.sort(axes))
            rank = len(in_shape)
            alphabet  = 'abcdefghijklmnopqrstuvwxyz'
            in1_str = alphabet[:axes[0]]
            in2_str = alphabet[axes[0]]
            ones_shape = (in_shape[axes[0]],)
            for i in range(len(axes)-1):
                in1_str += alphabet[axes[i] + 1 : axes[i + 1]]
                in2_str += alphabet[axes[i+1]]
                ones_shape += (in_shape[axes[i+1]],)
            in1_str += alphabet[axes[-1] + 1 : rank]
            self.einsum_str = '{},{}->{}'.format(
                in1_str,
                in2_str,
                alphabet[:rank],
                )
            self.ones_shape = ones_shape

    def compute_inline(self, x):
        rho = self.rho
        axes = self.axes

        if axes is None:
            x_max = np.max(x)
            smooth_max = x_max + 1/rho * np.log(np.sum(np.exp(rho * (x - x_max))))
            return smooth_max
        else:
            ones_shape = self.ones_shape
            axeswise_max = np.max(x, axis=self.axes)
            # print(self.einsum_str, axeswise_max.shape, ones_shape)
            difference = x - np.einsum(
                self.einsum_str,
                axeswise_max,
                np.ones(ones_shape),
                )
            exp = np.exp(rho * difference)
            summation = np.sum(exp, axis=axes)
            smooth_axeswise_max = axeswise_max + 1.0 / rho * np.log(summation)
            return smooth_axeswise_max

    def compute_jax(self, x):
        import jax.numpy as jnp
        rho = jnp.array(self.rho)
        axes = self.axes

        if axes is None:
            x_max = jnp.max(x)
            smooth_max = x_max + 1/rho * jnp.log(jnp.sum(jnp.exp(rho * (x - x_max))))
            return smooth_max
        else:
            ones_shape = self.ones_shape
            axeswise_max = jnp.max(x, axis=self.axes)
            # print(self.einsum_str, axeswise_max.shape, ones_shape)
            difference = x - jnp.einsum(
                self.einsum_str,
                axeswise_max,
                jnp.ones(ones_shape),
                )
            exp = jnp.exp(rho * difference)
            summation = jnp.sum(exp, axis=axes)
            smooth_axeswise_max = axeswise_max + 1.0 / rho * jnp.log(summation)
            return smooth_axeswise_max


    def evaluate_vjp(self, cotangents, x, y):
        if cotangents.check(x):
            if self.axes is None:
                rho = self.rho
                diff = x - y
                exp_x = csdl.exp(rho*diff)
                vjp = cotangents[y] * exp_x / csdl.sum(exp_x)
            else:
                rho = self.rho
                axes = self.axes

                in_str, out_str  = self.einsum_str.split('->')
                in_str, ones_str = in_str.split(',')
                exp_str = in_str + '->' + out_str
                exp_term = csdl.exp(rho*(x-csdl.expand(y, x.shape, exp_str)))
                sum = csdl.sum(exp_term, axes=axes)
                expanded_sum = csdl.expand(sum, out_shape=x.shape, action=exp_str)
                vjp = csdl.expand(cotangents[y] , x.shape, exp_str) * exp_term / expanded_sum

            cotangents.accumulate(x, vjp)

class ElementwiseMaximum(Operation):
    '''
    Elementwise maximum of all the Variables in the arguments.
    '''
    def __init__(self, *args, rho=20.):
        super().__init__(*args)
        self.name = 'elementwise_maximum'
        out_shapes = (args[0].shape,)
        self.rho   = rho
        self.set_dense_outputs(out_shapes)

    def compute_inline(self, *args):
        rho = self.rho
        ew_max = args[0]
        for arg in args[1:]:
            ew_max = np.maximum(ew_max, arg)

        summation = 0.
        for arg in args:
            summation += np.exp(rho * (arg - ew_max))

        smooth_ew_max = (ew_max + 1. / rho * np.log(summation))
        return smooth_ew_max

    def compute_jax(self, *args):
        import jax.numpy as jnp
        rho = jnp.array(self.rho)
        ew_max = args[0]
        for arg in args[1:]:
            ew_max = jnp.maximum(ew_max, arg)

        summation = 0.
        for arg in args:
            summation += jnp.exp(rho * (arg - ew_max))

        smooth_ew_max = (ew_max + 1. / rho * jnp.log(summation))
        return smooth_ew_max

    def evaluate_vjp(self, cotangents, *inputs_and_outputs):
        inputs = inputs_and_outputs[:len(self.inputs)]
        output = inputs_and_outputs[-1]
        rho = self.rho
        # sum  = cotangents[output]/csdl.sum(*[csdl.exp(rho*(arg-output)) for arg in inputs])
        for input_var in inputs:
            if cotangents.check(input_var):
                cotangents.accumulate(input_var, cotangents[output]/csdl.sum(*[csdl.exp(rho*(arg-input_var)) for arg in inputs]))

                # cotangents.accumulate(input_var, csdl.exp(rho*(input_var-output))*sum)

def maximum(*args, axes=None, rho=20.):
    '''
    Computes the maximum entry in the input tensor if a single argument is provided.
    Computes the maximum entries along the specified axes if `axes` argument is given.
    Computes the elementwise maximum of multiple variables of the same shape, 
    if multiple arguments are provided. Axes argument is not allowed in this case.

    Parameters
    ----------
    *args : tuple of Variable or np.ndarray objects
        Input tensor/s whose maximum needs to be computed.
    axes : tuple of int, default=None
        Axes along which to compute the maximum of the input tensor,
        if there's only one input tensor.
    rho : float, default=20.
        Smoothing parameter for the maximum function.

    Returns
    -------
    Variable
        Maximum entry in the input tensor if a single argument is provided.
        Maximum entries along the specified axes if `axes` argument is given.
        Elementwise maximum of multiple variables of the same shape, 
        if multiple arguments are provided.
    
    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x_val = np.arange(6).reshape(2,3)
    >>> x = csdl.Variable(value = x_val)
    >>> y1 = csdl.maximum(x)
    >>> y1.value
    array([5.])

    Maximum of a single tensor variable along a specified axis
    
    >>> y2 = csdl.maximum(x, axes=(1,))
    >>> y2.value
    array([2., 5.])

    Elementwise maximum of multiple tensor variables

    >>> y3 = csdl.maximum(x, 2 * np.ones((2,3)), np.ones((2,3)))
    >>> y3.value
    array([[2.        , 2.        , 2.03465736],
           [3.        , 4.        , 5.        ]])

    Note that `y3.value[0,2]` is not exactly `2.0` due to the smoothing term.
    It can be made closer to `2.0` by increasing the value of 
    the smoothing parameter rho as shown below.
    
    >>> y = csdl.maximum(x, 2 * np.ones((2,3)), np.ones((2,3)), rho=200)
    >>> y.value
    array([[2.        , 2.        , 2.00346574],
           [3.        , 4.        , 5.        ]])
    '''

    # Multiple Variables to find maximum
    if axes is not None and len(args) > 1:
        raise ValueError('Cannot find maximum of multiple Variables along specified axes. \
                         Use X = max(A,B,...) followed by out=max(X, axes=(...)) instead.')
    if any(args[i].shape != args[0].shape for i in range(1, len(args))):
        raise ValueError('All Variables must have the same shape.')
    
    # Single Variable to find maximum
    if axes is not None:
        if any(np.asarray(axes) > (len(args[0].shape)-1)):
            raise ValueError('Specified axes cannot be more than the rank of the Variable.')
        if any(np.asarray(axes) < 0):
            raise ValueError('Axes cannot have negative entries.')

    if len(args) == 1:
        if axes is None:
            out_shape = (1,)
        else:
            out_shape = tuple([x for i, x in enumerate(args[0].shape) if i not in axes])
            if len(out_shape) == 0:
                # raise ValueError('It is inefficient to find the maximum of a tensor Variable along all axes. \
                #                  Use maximum(A) to find the maximum of all tensor entries.')
                out_shape = (1,)
                axes = None
        op = Maximum(validate_and_variablize(args[0]), axes=axes, out_shape=out_shape, rho=rho)
    else:
        # axes is None for multiple variables
        args = [validate_and_variablize(x, raise_on_sparse=False) for x in args]
        op = ElementwiseMaximum(*args, rho=rho)
    
    return op.finalize_and_return_outputs()

class TestMaximum(csdl_tests.CSDLTest):
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = 3.0*np.arange(6).reshape(2,3)
        y_val = 2.0*np.ones((2,3))
        z_val = np.ones((2,3))
        d_val = np.arange(12).reshape(2,3,2)


        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        z = csdl.Variable(name = 'z', value = z_val)
        d = csdl.Variable(name = 'd', value = d_val)

        compare_values = []
        # maximum of a single tensor variable
        s1 = csdl.maximum(x)
        s1.add_name('s1')
        t1 = np.array([15.0])
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # maximum of a single tensor variable
        s1 = csdl.maximum(x, axes=(0,1))
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # maximum of a single tensor constant
        s2 = csdl.maximum(x_val)
        s2.add_name('s2')
        compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]

        # maximum of a single tensor variable along specified axes
        s3 = csdl.maximum(x, axes=(1,))
        t3 = np.max(x_val, axis=1)
        s3.add_name('s3')
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        # maximum of a single tensor variable along 2 specified axes
        s4 = csdl.maximum(d, axes=(0,2))
        t4 = np.max(d_val, axis=(0,2))
        s4.add_name('s4')
        compare_values += [csdl_tests.TestingPair(s4, t4, tag = 's4', decimal=8)]

        # elementwise maximum of multiple tensor variables
        s5 = csdl.maximum(x, y, z)
        t5 = np.maximum(x_val, y_val)
        # s5.add_name('s5')
        compare_values += [csdl_tests.TestingPair(s5, t5, tag = 's5', decimal=8)]

        # elementwise maximum of multiple tensor constants
        s6 = csdl.maximum(x_val, y_val, z_val)
        # s6.add_name('s6')
        compare_values += [csdl_tests.TestingPair(s6, t5, tag = 's6', decimal=8)]

        # maximum of a single tensor constant
        # compare_values = []
        n7 = np.array([10000.0,-10000.0])
        s7 = csdl.maximum(n7)
        s7.add_name('s7')
        t7 = np.max(n7).flatten()
        compare_values += [csdl_tests.TestingPair(s7, t7, tag = 's7')]

        # TODO: maximum of a zero tensor - need to check this 
        # to avoid errors from sum(log(1+1+..)) if there are multiple entries of zero
        # and zero is the maximum
        # zeros = np.zeros((2,3))
        # s7 = csdl.maximum(zeros, rho=2000)
        # t7 = np.array([0.0])
        # s7.add_name('s7')
        # compare_values += [csdl_tests.TestingPair(s7, t7, tag = 's7', decimal=3)]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_example(self,):
        self.docstest(maximum)

if __name__ == '__main__':
    test = TestMaximum()
    test.overwrite_backend = 'jax'
    test.test_functionality()
    test.test_example()