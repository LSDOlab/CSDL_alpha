import csdl_alpha as csdl
import numpy as np

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
        ct.loopify_subgraph(stacked_inputs, outputs)

def test_transform():

    recorder =csdl.Recorder(inline = True)
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
    z = x+x2

    da_transform = LoopTransformation(name = 'loop transformation')
    da_transform.transform(
        stacked_inputs = {
            a: csdl.Variable(value=np.array([[1.3, 1.5],[1.3, 1.5]])),
            b: csdl.Variable(value=np.array([[2.1, 2.0],[2.1, 2.0]])),
        },
        outputs = {x: None}
    )

    print()
    # f1.check_history()
    print()
    recorder.transformation_logger.print_history()
    recorder.transformation_logger.print_stack()
    print()
    print(da_transform.info())
    print(recorder.transformation_logger.get_current().info())
    exit()
    
if __name__ == '__main__':
    test_transform()