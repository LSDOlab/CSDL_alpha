from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation

class Add(ElementwiseOperation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'add'

    def compute_inline(self, x, y):
        return x + y

    def evaluate_jacobian(self, x, y):
        return csdl.Constant(x.size, val = 1), csdl.Constant(y.size, val = 1)

    def evaluate_jvp(self, x, y, vx, vy):
        return add(vx.flatten(), vy.flatten())

    def evaluate_vjp(self, x, y, vout):
        return vout.flatten(), vout.flatten()

class BroadcastAdd(ElementwiseOperation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'add'

    def compute_inline(self, x, y):
        return x.value + y.value

    def evaluate_jacobian(self, x, y):
        return csdl.Constant(x.size, val = 1), csdl.Constant(y.size, val = 1)

    def evaluate_jvp(self, x, y, vx, vy):
        return add(vx.flatten(), vy.flatten())

    def evaluate_vjp(self, x, y, vout):
        return vout.flatten(), vout.flatten()

class SparseAdd(ElementwiseOperation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'add'

    def compute_inline(self, x, y):
        return x.value + y.value

    def evaluate_jacobian(self, x, y):
        return csdl.Constant(x.size, val = 1), csdl.Constant(y.size, val = 1)

    def evaluate_jvp(self, x, y, vx, vy):
        return add(vx.flatten(), vy.flatten())

    def evaluate_vjp(self, x, y, vout):
        return vout.flatten(), vout.flatten()
    
class BroadcastSparseAdd(ElementwiseOperation):

    def __init__(self,x,y):
        super().__init__(x,y)
        self.name = 'add'

    def compute_inline(self, x, y):
        return x.value + y.value

    def evaluate_jacobian(self, x, y):
        return csdl.Constant(x.size, val = 1), csdl.Constant(y.size, val = 1)

    def evaluate_jvp(self, x, y, vx, vy):
        return add(vx.flatten(), vy.flatten())

    def evaluate_vjp(self, x, y, vout):
        return vout.flatten(), vout.flatten()

def add(x,y):
    """
    doc strings
    """

    op = Add(x,y)
    return op.finalize_and_return_outputs()


class TestAdd():
    def test_functionality(self,):
        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x = csdl.Variable(name = 'x', value = 3.0)
        y = csdl.Variable(name = 'y', value = 2.0)

        z = csdl.add(x,y)

        assert z.value == np.array([5.])
        assert z.shape == (1,)

    def test_example1(self,):
        from numpy.testing import assert_array_equal

        # docs: entry
        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x = csdl.Variable(name = 'x', value = np.ones((3,2))*3.0)
        y = csdl.Variable(name = 'z', value = np.ones((3,2))*2.0)
        z = csdl.add(x,y)
        print(z.value)
        # docs: exit

        assert_array_equal(z.value, np.ones((3,2))*5.0)

if __name__ == '__main__':
    test = TestAdd()
    test.test_functionality()
    test.test_example1()