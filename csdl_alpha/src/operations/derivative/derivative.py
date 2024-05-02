from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.graph import Graph
from csdl_alpha.utils.inputs import variablize, validate_and_variablize
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.utils.typing import VariableLike

from typing import Union

class Reverse():

    def __init__(self,x:Variable,y:Variable):
        super().__init__(x,y)
        self.name = 'reverse'

    def compute_inline(self, x, y):
        return x + y

    def compute_inline(self, x, y):
        return x + y

class VarTangents():
    def __init__(self):
        """Store cotangents/tangents for variables in the graph.
        """
        self.tangent_dictionary:dict[Variable: Variable] = {}

    def accumulate(self, variable:Variable, tangent:Variable)->None:
        """Accumulate a tangent for a variable.

        Args:
            variable (Variable): The variable to accumulate the tangent for.
            tangent (Variable): The tangent to accumulate.
        """
        if variable.shape != tangent.shape:
            raise ValueError(f"Variable shape {variable.shape} and (co)tangent shape {tangent.shape} do not match.")
        if variable in self.tangent_dictionary:
            if self.tangent_dictionary[variable] is None:
                self.tangent_dictionary[variable] = tangent
            elif isinstance(tangent, Variable):
                self.tangent_dictionary[variable] = self.tangent_dictionary[variable] + tangent
            else:
                raise TypeError(f"Expected tangent to be a Variable, but got {type(tangent)}.")
        else:
            raise KeyError(f"Variable {variable} not found in tangent dictionary.")

        self.tangent_dictionary[variable].add_name(f'tangent_{variable.name}')

    def initialize(self, variable:Variable):
        """Initialize the tangent for a variable.

        Args:
            variable (Variable): The variable to initialize the tangent for.
        """
        if variable not in self.tangent_dictionary:
            import numpy as np
            self.tangent_dictionary[variable] = None
            # self.tangent_dictionary[variable].add_name(f'tangent_{variable.name}')

    def check(self, variable:Variable):
        """Check if a variable has a tangent.

        Args:
            variable (Variable): The variable to check.

        Returns:
            bool: True if the variable has a tangent.
        """
        return variable in self.tangent_dictionary

    def __getitem__(self, variable:Variable):
        """Get the tangent for a variable.

        Args:
            variable (Variable): The variable to get the tangent for.

        Returns:
            Variable: The tangent for the variable.
        """
        return self.tangent_dictionary[variable]


def build_derivative_node_order(
        graph:Graph,
        ofs: list[Variable],
        wrts: list[Variable],
        reverse: bool = True,
    )->list[Variable]:
    """
    Gets the subgraph of the graph that is relevant for the derivative of the output variables with respect to the input variables.
    """
    
    import rustworkx as rx
    
    # Find the subgraph of ofs, wrts
    intersecting_nodes = graph._get_intersection(
        wrts,
        ofs,
        check_sources=False,
        check_targets=False,
        add_hanging_input_variables=False,
        add_hanging_output_variables=False,
    )
    for of_var in ofs:
        intersecting_nodes.add(graph.node_table[of_var])
    for wrt_var in wrts:
        intersecting_nodes.add(graph.node_table[wrt_var])
    # descendants = set()
    # wrt_var_indices = set()
    # for wrt_var in wrts:
    #     descendants.update(rx.descendants(graph.rxgraph, graph.node_table[wrt_var]))
    #     wrt_var_indices.add(graph.node_table[wrt_var])
    # relevant_nodes = rx.ancestors(graph.rxgraph, graph.node_table[of_var]).intersection(descendants)
    # relevant_nodes.add(graph.node_table[of_var])
    # relevant_nodes.update(wrt_var_indices)

    # Compute the order of nodes to process
    node_order = []
    for node in reversed(rx.topological_sort(graph.rxgraph)):
        node_index = node
        if node_index in intersecting_nodes:
            node_order.append(graph.rxgraph[node])
    return node_order

def listify_and_verify_variables(variables:Union[Variable, list[Variable]])->list[Variable]:
    if isinstance(variables, (list, tuple)):
        variables = [validate_and_variablize(var) for var in variables]
    else:
        variables = [validate_and_variablize(variables)]
    return variables

def vjp(seeds:list[tuple[Variable, Variable]],
        wrts:Union[Variable, list[Variable]],
        graph:Graph,
    )->dict[Variable]:
    """ Computes the vector-Jacobian product of the seeds with respect to the wrts in the graph.

    Parameters
    ----------
    seeds : list[tuple[Variable, Variable]]
        A list of variable and seed pairs
    wrts : Union[Variable, list[Variable]]
        A list of variables to propagate derivatives through
    graph : Graph
        The graph in which to compute the derivatives

    Returns
    -------
    dict[Variable]
        The accumulated cotangents for the wrt variables given the output seeds

    Raises
    ------
    ValueError
        Seeds must match the shape of the associated variable
    """

    # Preprocess inputs
    of_vars = []
    for of_var, of_seeds in seeds:
        of_vars.append(validate_and_variablize(of_var))

        # Seeds must match shape of the variable
        if of_seeds.shape != of_var.shape:
            raise ValueError(f"Seed shape {of_seeds.shape} and variable shape {of_var.shape} do not match.")
    wrt_vars = listify_and_verify_variables(wrts)

    # Extract subgraph of relevant nodes in the graph
    node_order = build_derivative_node_order(graph, of_vars, wrt_vars)

    # initialize seeds and final wrt cotangents
    import numpy as np
    import csdl_alpha as csdl

    cotangents = VarTangents()
    for i, of_var in enumerate(of_vars):
        cotangents.initialize(of_var)
        cotangents.accumulate(of_var, seeds[i][1])

    # perform the vector-jacobian here in terms of CSDL operations by going through the node order
    for node in node_order:
        if isinstance(node, Variable):
            cotangents.initialize(node)

    for node in node_order:
        # rec = csdl.get_current_recorder()
        # rec.visualize_graph(filename = graph.name)
        if isinstance(node, Operation):
            # Moved to composed operation class for now
            # for output in node.outputs:
            #     if cotangents[output] is None:
            #         cotangents.accumulate(output, csdl.Variable(value = np.zeros(output.shape)))

            node.evaluate_vjp(cotangents, *node.inputs, *node.outputs)

    wrt_cotangents:dict[Variable:Variable] = {}
    for wrt_var in wrt_vars:
        wrt_cotangent = cotangents[wrt_var]
        if wrt_cotangent is None:
            wrt_cotangents[wrt_var] = None
            continue
            # wrt_cotangent = csdl.Variable(name = f'seeds_{wrt_var.name}', value = np.zeros(wrt_var.shape))
        wrt_cotangents[wrt_var] = wrt_cotangent
    return wrt_cotangents

def reverse(of: Variable, wrts: Union[Variable, list[Variable]])->dict[Variable]:
    of_var = validate_and_variablize(of)
    wrt_vars = listify_and_verify_variables(wrts)
    
    import csdl_alpha as csdl
    graph = csdl.get_current_recorder().active_graph

    # Node order built in VJP for now..
    # node_order = build_derivative_node_order(graph, [of_var], wrt_vars)

    # perform the reverse mode differentiation:
    import numpy as np
    jacobians:dict[Variable:Variable] = {}
    for wrt_var in wrt_vars:
        jacobians[wrt_var] = csdl.Variable(name = f'jacobian_{of.name}_wrt_{wrt_var.name}', value = np.zeros((of_var.size, wrt_var.size)))

    initial_output_seed = csdl.Variable(name = f'seed_{of.name}', value = np.zeros(of_var.size))
    # for row_index in csdl.frange(of_var.size):
    for row_index in range(of_var.size):
        current_output_seed = initial_output_seed.set(csdl.slice[row_index], 1.0)
        current_output_seed = current_output_seed.reshape(of_var.shape)

        #TODO: pass in node order first somehow. Right now, we are 
        vjp_cotangents = vjp([(of_var,current_output_seed)], wrt_vars, graph)

        for wrt_var in wrt_vars:
            wrt_cotangent = vjp_cotangents[wrt_var]
            if wrt_cotangent is None:
                continue
            jacobians[wrt_var] = jacobians[wrt_var].set(csdl.slice[row_index, :], wrt_cotangent.flatten())
            jacobians[wrt_var].add_name(f'jacobian_{of.name}_wrt_{wrt_var.name}')
        
        # OLD:
        # current_output_seed = initial_output_seed.set(csdl.slice[row_index], 1.0)
        # current_output_seed = current_output_seed.reshape(of_var.shape)
        # cotangents = VarTangents()
        # cotangents.initialize(of_var)
        # cotangents.accumulate(of_var, current_output_seed)

        # for node in node_order:
        #     if isinstance(node, Variable):
        #         cotangents.initialize(node)

        # for node in node_order:
        #     if isinstance(node, Operation):
        #         node.evaluate_vjp(cotangents, *node.inputs, *node.outputs)

    return jacobians

class TestDerivative(csdl_tests.CSDLTest):
    
    def test_functionality_scalar(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = 3.0
        y_val = 4.0
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        
        xy = x*y
        xy.add_name('xy')
        z = xy+x
        z.add_name('z2')

        dz_dx = csdl.derivative.reverse(of = z, wrts = x)[x]
        dz_dx.add_name('dz_dx')
        print(dz_dx.value) 
        print((y+1.0).value) 

        dz2_dxy = csdl.derivative.reverse(of = dz_dx, wrts = y)[y]
        print(dz2_dxy.value) # 1.0

        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(format = 'png')

    def test_functionality_vector(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np
        x_val = np.array([[3.0, 4.0]])
        y_val = np.array([[5.0, 6.0]])
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        
        xy = x*y
        xy.add_name('xy')
        z = xy+x
        z.add_name('z2')

        dz_dx = csdl.derivative.reverse(of = z, wrts = x)[x]
        dz_dx.add_name('dz_dx')
        # print(dz_dx.value) 
        # print(np.diagflat((y+1.0).value)) 

        dz2_dxy = csdl.derivative.reverse(of = dz_dx, wrts = y)[y]
        # print(dz2_dxy.value) # 1.0

        # recorder = csdl.get_current_recorder()
        # recorder.visualize_graph(format = 'png')


    def test_expand(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = 3.0
        y_val = np.array([1.0, 2.0, 3.0])

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)

        compare_values = []
        # expand a scalar constant
        # s1 = csdl.expand(x_val, out_shape=(2,3,4))
        # t1 = x_val * np.ones((2,3,4))
        # compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]

        # expand a vector
        s3 = csdl.expand(y, out_shape=(3,4), action='j->jk')
        t3 = np.einsum('j,jk->jk', y_val, np.ones((3,4)))
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_max_elementwise(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()

        x_val = 3.0*np.arange(6).reshape(2,3)
        y_val = 2.0*np.ones((2,3))
        z_val = np.ones((2,3))
        d_val = np.arange(12).reshape(2,3,2)
        t1 = np.array([15.0])

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)
        z = csdl.Variable(name = 'z', value = z_val)

        compare_values = []

        # elementwise maximum of multiple tensor variables
        # s5 = csdl.maximum(x, y, z)
        # t5 = np.maximum(x_val, y_val)
        # compare_values += [csdl_tests.TestingPair(s5, t5, tag = 's5', decimal=8)]

        # s2 = csdl.maximum(x_val)
        # compare_values += [csdl_tests.TestingPair(s2, t1, tag = 's2')]

        s3 = csdl.maximum(x, axes=(1,))
        t3 = np.max(x_val, axis=1)
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]


        self.run_tests(compare_values = compare_values, verify_derivatives=True)


    def test_avg(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        x_val = 3.0*np.arange(6).reshape(2,3)
        y_val = 2.0*np.ones((2,3))
        z_val = np.ones((2,3))

        x = csdl.Variable(name = 'x', value = x_val)
        # y = csdl.Variable(name = 'y', value = y_val)
        # z = csdl.Variable(name = 'z', value = z_val)

        compare_values = []
        # average of a single tensor variable
        # s1 = csdl.average(x)
        # t1 = np.array([7.5])
        # compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's1')]


        x_val = 3.0*np.arange(24).reshape(2,3,4)
        x = csdl.Variable(value = x_val)
        # average of a single tensor variable along specified axes
        s3 = csdl.average(x, axes=(1,2))
        t3 = np.average(x_val, axis=(1,2))
        compare_values += [csdl_tests.TestingPair(s3, t3, tag = 's3')]
        print(csdl.derivative.reverse(of = s3, wrts = x)[x].value)

        self.run_tests(compare_values = compare_values, verify_derivatives=True)


    def test_linear2(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        A_shape = (2,2)
        b_shape = (2,1)
        A_val = (np.arange(np.prod(A_shape)).reshape(A_shape)+1.0)**2.0
        b_val = np.arange(np.prod(b_shape)).reshape(b_shape)
        b = csdl.Variable(value = b_val, name = 'b')
        A = csdl.Variable(value = A_val, name = 'A')

        compare_values = []
        x = csdl.solve_linear(A,b)
        deriv = csdl.derivative.reverse(of = x, wrts = [b])[b]
        print(deriv.value)
        compare_values += [csdl_tests.TestingPair(x, np.linalg.solve(A_val, b_val))]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)


    def test_linear(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        n = 3
        # A_shape = (n,n)
        # b_shape = (n,1)
        # # Try one: doesnt work... Condition number is too high?
        # A_val = (np.arange(np.prod(A_shape)).reshape(A_shape)+1)**2.0
        # b_val = np.arange(np.prod(b_shape)).reshape(b_shape)
        
        # Try two: works...
        # A_val = np.diagflat(np.ones(n)*3.0)
        # A_val[0,1] = 2.0
        # A_val[1,0] = 2.0
        # b_val = np.arange(n).reshape(b_shape)

        # Try three: works...
        main_diag = np.arange(n)+1
        A_val = np.diag(main_diag) + np.diag(main_diag[:-1]+1, 1) + np.diag(main_diag[:-1]+2, -1)
        b_val = 2*np.arange(n)

        b = csdl.Variable(value = b_val, name = 'b')
        A = csdl.Variable(value = A_val, name = 'A')

        compare_values = []
        x = csdl.solve_linear(A,b)
        compare_values += [csdl_tests.TestingPair(x, np.linalg.solve(A_val, b_val))]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def test_multi_get(self):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        compare_values = []

        x_val = np.array([1.0, 2.0, 3.0])
        x = csdl.Variable(name = 'x', value = x_val)

        def func(x):
            return  x[[0, 1]]
        y = func(x)

        compare_values += [csdl_tests.TestingPair(y, x_val[[0, 1]])]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)


    def test_composed(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()
        compare_values = []

        x_val = 3.0
        y_val = 2.0
        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)

        s1 = csdl.sub(x,y)
        s1.add_name('s1')
        t1 = np.array([x_val - y_val])
        # csdl.derivative.reverse(of = s1, wrts = [x,y])
        compare_values += [csdl_tests.TestingPair(s1, t1, tag = 's3')]
        self.run_tests(compare_values = compare_values, verify_derivatives=True)


    def test_division(self,):
        self.prep()

        import csdl_alpha as csdl
        import numpy as np

        recorder = csdl.build_new_recorder(inline = True)
        recorder.start()

        x_val = np.arange(10).reshape((2,5))+1.0
        y_val = np.arange(10).reshape((2,5))*0.5+1.0

        x = csdl.Variable(name = 'x', value = x_val)
        y = csdl.Variable(name = 'y', value = y_val)

        compare_values = []

        # Variables:
        z = csdl.div(x,y)
        compare_values += [csdl_tests.TestingPair(z, x_val/y_val)]

        z = csdl.div(x[0,0],y)
        compare_values += [csdl_tests.TestingPair(z, x_val[0,0]/y_val)]

        z = csdl.div(x,y[0,0])
        compare_values += [csdl_tests.TestingPair(z, x_val/y_val[0,0])]
        self.run_tests(compare_values = compare_values, verify_derivatives=True)

    def issue(self,):
        # Define variables: using openmdao solved optimization values
        import csdl_alpha as csdl
        z1 = csdl.Variable(name = 'z1', value = 1.97763888)
        z2 = csdl.Variable(name = 'z2', value = 8.83056605e-15)
        x = csdl.Variable(name = 'x', value = 0.0)
        y2 = csdl.ImplicitVariable(name = 'y2', value = 1.0)

        # Define each "component" from the example
        with csdl.namespace('Discipline 1'):
            y1 = z1**2 + z2 + x - 0.2*y2
            y1.add_name('y1')

        with csdl.namespace('Discipline 2'):
            residual = csdl.sqrt(y1) + z1 + z2 - y2
            residual.add_name('residual')

        # Specifiy coupling
        with csdl.namespace('Couple'):
            solver = csdl.nonlinear_solvers.GaussSeidel()
            solver.add_state(y2, residual, state_update=y2+residual, tolerance=1e-8)
            solver.run()

        dy1dx = csdl.derivative.reverse(of = y1, wrt = x) # What should this value be???? Should this go through the nonlinear solver?? or stay inside the residual function???
        # dy1dx is not part of the residual function graph so it should be IFTed

    # def test_docstring(self):
    #     self.docstest(add)

if __name__ == '__main__':
    test = TestDerivative()
    test.test_functionality_scalar()
    test.test_functionality_vector()
    test.test_expand()
    test.test_max_elementwise()
    test.test_composed()
    test.test_division()
    test.test_avg()
    test.test_log()
    test.test_linear()
    test.test_multi_get()