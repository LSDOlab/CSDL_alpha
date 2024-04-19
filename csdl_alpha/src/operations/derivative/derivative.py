from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.utils.typing import VariableLike

class Reverse():

    def __init__(self,x:Variable,y:Variable):
        super().__init__(x,y)
        self.name = 'reverse'

    def compute_inline(self, x, y):
        return x + y

    def compute_inline(self, x, y):
        return x + y

class VarSeeds():
    def __init__(self, x: Variable):
        self.x = x

    def set(self, slice, value):
        return self

def reverse(of: Variable, wrt: Variable)->Variable:
    of_var = validate_and_variablize(of, raise_on_sparse=False)
    wrt_var = validate_and_variablize(wrt, raise_on_sparse=False)

    import csdl_alpha as csdl
    graph = csdl.get_current_recorder().active_graph

    import rustworkx as rx

    # Compute the order of nodes to process
    node_order = []
    relevant_nodes = rx.ancestors(graph.rxgraph, graph.node_table[of_var]).intersection(rx.descendants(graph.rxgraph, graph.node_table[wrt_var]))
    relevant_nodes.update(set([graph.node_table[wrt_var], graph.node_table[of_var]]))
    for node in reversed(rx.topological_sort(graph.rxgraph)):
        if node in relevant_nodes:
            node_order.append(graph.rxgraph[node])
    print(node_order)


    # perform the reverse mode differentiation:
    import numpy as np
    for row_index in range(of_var.size):
        current_seed = csdl.Variable(name = f'seed_{row_index}', value = np.zeros(of_var.size))
        current_seed = current_seed.set(csdl.slice[row_index], 1.0)
        var_seeds = {of_var: current_seed}

        for node in node_order:
            if isinstance(node, csdl.Variable):
                continue

            node.evaluate_jvp(*node.inputs, var_seeds)
            current_seed = node.reverse(current_seed)

class TestDerivative(csdl_tests.CSDLTest):
    
    def test_functionality(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = 3.0
        y_val = 2.0
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        
        z = x*y+x
        z.add_name('z')

        dz_dx = csdl.derivative.reverse(of = z, wrt = x)

        recorder = csdl.get_current_recorder()
        recorder.visualize_graph()

    def test_docstring(self):
        self.docstest(add)

if __name__ == '__main__':
    test = TestDerivative()
    test.test_functionality()
    # test.test_errors()