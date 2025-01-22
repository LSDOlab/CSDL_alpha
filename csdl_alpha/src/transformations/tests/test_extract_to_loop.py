import csdl_alpha as csdl
import numpy as np
import pytest

def f(x1, x2):
    z = x1*x2
    A = csdl.expand(x1, (3,2))
    b = csdl.expand(x2, (2,3))
    d = csdl.sin(z)*A
    return d, A@b

class LoopTransformation(csdl.transforms.TransformationBase):

    def post_init(self):
        self.metadata = {}

    def apply(self, stacked_inputs, outputs):
        recorder = self.get_current_recorder()
        graph = recorder.active_graph

        ct = csdl.transformation_helper
        return ct.loopify_subgraph(stacked_inputs, outputs)

def test_transform():

    recorder =csdl.Recorder(inline = False, debug=True)
    recorder.start()
    a = csdl.Variable(value=np.array([1.3, 1.5]))
    b = csdl.Variable(value=np.array([2.1, 2.0]))
    c = csdl.Variable(value=-1)
    x = csdl.ImplicitVariable(value=np.array([0.34, 0.30])) # solution:[0.38462, 0.38743]
    x2 = csdl.Variable(value=np.array([0.34, 0.30])) # solution:[0.38462, 0.38743]

    ax2 = a*((x+x2)/2)**2
    y = x - (-ax2 - c)/b
    y2 = x2 - (-ax2 - c)/(2*b)

    solver = csdl.nonlinear_solvers.Newton(print_status=True)
    solver.add_state(x, y)
    solver.add_state(x2, y2)
    solver.run()
    z = x+x2 + a

    a_perturb = csdl.Variable(value=0.1, name = 'a_perturb')
    stack_a = csdl.Variable(value=np.array([[1.3, 1.5],[1.35, 1.55]])) + a_perturb
    stack_b = csdl.Variable(value=np.array([[2.1, 2.0],[2.3, 2.4]]))
    da_transform = LoopTransformation(name = 'loop transformation')
    output_stacks = da_transform.transform(
        stacked_inputs = {
            a: stack_a,
            b: stack_b,
        },
        outputs = {x: None}
    )
    
    output_stacks[x].name = ('intermediate stack')
    da_transform = LoopTransformation(name = 'loop transformation2')
    output_stacks = da_transform.transform(
        stacked_inputs = {
            x2: output_stacks[x]+1.0,
        },
        outputs = {z: None}
    )
    z_stacked = output_stacks[z]
    stacked_a_perturb = csdl.Variable(value=np.array([[0.1], [0.2], [0.3]]))
    da_transform = LoopTransformation(name = 'loop transformation3')
    output_stacks = da_transform.transform(
        stacked_inputs = {
            a_perturb: stacked_a_perturb,
        },
        outputs = {z_stacked: None}
    )

    # recorder.visualize_graph('2', visualize_style='hierarchical')
    derivs = csdl.derivative(output_stacks[z_stacked], stacked_a_perturb)
    recorder.execute()

    real = np.array(
        [[[3.24522405, 3.43556376],
        [3.21562852, 3.3802278 ]],

        [[3.33901864, 3.52960719],
        [3.31006835, 3.47541554]],

        [[3.43299235, 3.62381823],
        [3.40465766, 3.57072072]]],
    )
    np.testing.assert_allclose(output_stacks[z_stacked].value, real, rtol=1e-5)

def test_transform2():

    def f10(x):
        return x**2.0
    
    def f01(y):
        return csdl.sin(y+0.5)**2.0
    
    def f11(x, y):
        return csdl.average(csdl.outer(x,y))
    
    recorder =csdl.Recorder(inline = False, debug=True)
    recorder.start()    
    x = csdl.Variable(value=np.array([1.0]))
    y = csdl.Variable(value=np.array([1.0]))

    x_out = f10(x)
    y_out = f01(y)
    z = f11(x_out, y_out)

    # Do the loop thing
    # x
    x_stacked = csdl.Variable(value=np.array([[1.0],[2.0]]))
    x_out_stacked = LoopTransformation().transform({x:x_stacked},{x_out:None})[x_out]

    # y
    y_stacked = csdl.Variable(value=np.array([[3.0],[4.0]]))
    y_out_stacked = LoopTransformation().transform({y:y_stacked},{y_out:None})[y_out]

    # z
    x_out_stacked2 = csdl.expand(x_out_stacked, (2,1,2), 'ij->ijk').reshape(4,1)
    y_out_stacked2 = csdl.expand(y_out_stacked, (2,2,1), 'ij->kij').reshape(4,1)
    z_stacked = LoopTransformation().transform({x_out:x_out_stacked2, y_out:y_out_stacked2},{z:None})[z]
    # recorder.visualize_graph(visualize_style='hierarchical')
    recorder.execute()

    # Real calculation:
    def np_f10(x):
        return x**2.0
    
    def np_f01(y):
        return np.sin(y+0.5)**2.0
    
    def np_f11(x, y):
        return np.average(np.outer(x,y))
    
    x_stacked_np = np.array([[1.0],[2.0]])
    x_out_stacked_np = np_f10(x_stacked_np)
    y_stacked_np = np.array([[3.0],[4.0]])
    y_out_stacked_np = np_f01(y_stacked_np)
    x_out_stacked2_np = np.repeat(x_out_stacked_np, 2, axis=0).reshape(4,1)
    y_out_stacked2_np = np.repeat(y_out_stacked_np, 2, axis=1).ravel(order='F').reshape(4,1)
    z_stacked_np = []
    for i in range(4):
        z_np = np_f11(x_out_stacked2_np[i], y_out_stacked2_np[i])
        z_stacked_np.append(z_np)
    z_stacked_np = np.array(z_stacked_np).reshape(4,1)

    # check all values:
    np.testing.assert_allclose(x_out_stacked.value, x_out_stacked_np, rtol=1e-5)
    np.testing.assert_allclose(y_out_stacked.value, y_out_stacked_np, rtol=1e-5)
    np.testing.assert_allclose(x_out_stacked2.value, x_out_stacked2_np, rtol=1e-5)
    np.testing.assert_allclose(y_out_stacked2.value, y_out_stacked2_np, rtol=1e-5)
    np.testing.assert_allclose(z_stacked.value, z_stacked_np, rtol=1e-5)



def test_errors():

    def f10(x):
        return x**2.0
    
    def f01(y):
        return csdl.sin(y+0.5)**2.0
    
    def f11(x, y):
        return csdl.average(csdl.outer(x,y))
    
    recorder =csdl.Recorder(inline = False, debug=True)
    recorder.start()    
    x = csdl.Variable(value=np.array([1.0]))
    y = csdl.Variable(value=np.array([1.0]))

    x_out = f10(x)
    y_out = f01(y)
    z = f11(x_out, y_out)

    # Do the loop thing
    # x
    with pytest.raises(ValueError):
        x_stacked = csdl.Variable(value=np.array([[1.0, 1.5],[2.0, 1.5]]))
        x_out_stacked = LoopTransformation().transform({x:x_stacked},{x_out:None})[x_out]

    with pytest.raises(ValueError):
        x_stacked = csdl.Variable(value=np.array([[1.0],[2.0]]))
        y_stacked = csdl.Variable(value=np.array([[1.0],[2.0],[3.0]]))
        x_out_stacked = LoopTransformation().transform({x:x_stacked, y:y_stacked},{x_out:None})[x_out]

    # recorder.visualize_graph(visualize_style='hierarchical')
    recorder.execute()

if __name__ == '__main__':
    # test_transform()
    # test_transform2()
    test_errors()