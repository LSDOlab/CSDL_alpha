from csdl_alpha.src.operations.operation_subclasses import ElementwiseOperation, ComposedOperation
from csdl_alpha.src.operations.derivatives.reverse import _vjp, preprocess_reverse
from csdl_alpha.src.operations.derivatives.bookkeeping import listify_and_verify_variables
from csdl_alpha.src.graph.operation import Operation, set_properties 
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.graph import Graph
from csdl_alpha.utils.inputs import variablize, validate_and_variablize, get_type_string
import csdl_alpha.utils.testing_utils as csdl_tests
from csdl_alpha.utils.typing import VariableLike


from typing import Union



def reverse(
        of: Variable,
        wrts: Union[Variable, list[Variable]],
        graph:Graph = None,
        loop:bool = True,
        elementwise = False,
    )->dict[Variable]:
    of_var = validate_and_variablize(of)
    wrt_vars = listify_and_verify_variables(wrts)
    
    import csdl_alpha as csdl

    if graph is None:
        graph = csdl.get_current_recorder().active_graph
    else:
        if not isinstance(graph, Graph):
            raise TypeError(f"Expected graph to be a Graph, but got {get_type_string(graph)}.")
    # Node order built in VJP for now..
    # node_order = build_derivative_node_order(graph, [of_var], wrt_vars)
    node_order = preprocess_reverse([of_var], wrt_vars, graph)

    # perform the reverse mode differentiation:
    import numpy as np
    jacobians:dict[Variable:Variable] = {}
    for wrt_var in wrt_vars:
        jacobians[wrt_var] = csdl.Variable(name = f'jac_{of.name}_wrt_{wrt_var.name}', value = np.zeros((of_var.size, wrt_var.size)))

    initial_output_seed = csdl.Variable(name = f'seed_{of.name}', value = np.zeros(of_var.size))
    
    if not elementwise:
        if loop:
            # Assume derivatives do not stack by default. Possible make this an option in the future.
            loop_d = csdl.frange(of_var.size, inline_lazy_stack=True)
        else:
            loop_d = range(of_var.size)
        for row_index in loop_d:
            current_output_seed = initial_output_seed.set(csdl.slice[row_index], 1.0)
            current_output_seed = current_output_seed.reshape(of_var.shape)

            #TODO: pass in node order first somehow. Right now, we are 
            # vjp_cotangents = vjp([(of_var,current_output_seed)], wrt_vars, graph)
            vjp_cotangents = _vjp([(of_var,current_output_seed)], wrt_vars, node_order)

            for wrt_var in wrt_vars:
                wrt_cotangent = vjp_cotangents[wrt_var]
                if wrt_cotangent is None:
                    continue
                jacobians[wrt_var] = jacobians[wrt_var].set(csdl.slice[row_index, :], wrt_cotangent.flatten())
                jacobians[wrt_var].add_name(f'jac_{of.name}_wrt_{wrt_var.name}')
        if loop:
            loop_d.op.name = 'r_loop'
    else:
        current_output_seed = csdl.Variable(name = f'seed_{of.name}', value = np.ones(of_var.shape))
        # vjp_cotangents = vjp([(of_var,current_output_seed)], wrt_vars, graph)
        vjp_cotangents = _vjp([(of_var,current_output_seed)], wrt_vars, node_order)

        for wrt_var in wrt_vars:
            wrt_cotangent = vjp_cotangents[wrt_var]
            if wrt_cotangent is None:
                continue
            diag_indices = list(np.arange(wrt_var.size))
            jacobians[wrt_var] = jacobians[wrt_var].set(csdl.slice[diag_indices, diag_indices], wrt_cotangent.flatten())
            jacobians[wrt_var].add_name(f'jac_{of.name}_wrt_{wrt_var.name}_diag')

    return jacobians


def derivative(
    ofs:Union[Variable, list[Variable]],
    wrts:Union[Variable, list[Variable]],
    mode:str = 'reverse',
    as_block:bool = False,
    graph:Graph = None,
    loop:bool = True,
    elementwise = False,
    )->Union[dict[Variable], dict[Variable,Variable], Variable]:
    """Computes the derivatives of the output variables with respect to the input variables in CSDL.

    Parameters
    ----------
    ofs : Union[Variable, list[Variable]]
        Variables to take derivatives of.
    wrts : Union[Variable, list[Variable]]
        Variables to take derivatives with respect to.
    mode : str, optional
        'forward' or 'reverse' to forward or reverse mode differentiation, by default 'reverse'
    as_block : bool, optional
        If True, returns the derivatives as a block matrix, by default False
    graph : Graph, optional
        Which graph to take derivatives of, by default the current active graph
    loop : bool, optional
        If True, uses a csdl loop to compute the derivatives, by default True
    elementwise : bool, optional
        If True, assumes diagonal derivatives, by default False

    Returns
    -------
    Union[dict[Variable], dict[Variable,Variable], Variable]
        Returns the derivatives as CSDL variables. 
        - If both ofs and wrts are lists, returns a dictionary of dictionaries.
        - If only one is a list, returns a dictionary.
        - If neither are lists, returns a single variable.

    Examples
    --------
    >>> recorder = csdl.Recorder(inline = True)
    >>> recorder.start()
    >>> x = csdl.Variable(value = 3.0)
    >>> y = csdl.Variable(value = 4.0)
    >>> z = x*y
    >>> dz = csdl.derivative(ofs = z, wrts = [x, y])
    >>> dz_dx, dz_dy = dz[x], dz[y]
    >>> dz_dx.value
    array([[4.]])
    >>> dz_dy.value
    array([[3.]])

    Take derivatives of derivatives

    >>> dz2_dx2 = csdl.derivative(ofs = dz_dx, wrts = x)
    >>> dz2_dx2.value
    array([[0.]])
    """
    of_is_list = True
    wrt_is_list = True
    if not isinstance(ofs, (list, tuple)):
        ofs = [ofs]
        of_is_list = False
    if not isinstance(wrts, (list, tuple)):
        wrts = [wrts]
        wrt_is_list = False

    if elementwise:
        first_var_size = ofs[0].size
        for var in ofs + wrts:
            if var.size != first_var_size:
                raise ValueError(f"Elementwise option requires all derivative variables to have the same size. Got size {var.size}, expected {first_var_size}.")

    if mode == 'reverse':
        output_dict = {}
        for of in ofs:
            deriv_of = reverse(of, wrts, graph, loop = loop, elementwise = elementwise)
            for wrt in deriv_of:
                output_dict[of, wrt] = deriv_of[wrt]
    elif mode == 'forward':
        raise NotImplementedError("Forward mode not implemented yet.")
    else:
        raise ValueError(f"Derivative mode {mode} not recognized.")

    if as_block:
        if len(ofs) == 1 and len(wrts) == 1:
            return output_dict[ofs[0], wrts[0]]
        else:
            block_mat_list = []
            for of in ofs:
                row_list = []
                for wrt in wrts:
                    row_list.append(output_dict[of, wrt])
                block_mat_list.append(row_list)

            from csdl_alpha.src.operations.linalg.blockmat import blockmat
            return blockmat(block_mat_list)

    if of_is_list and wrt_is_list:
        return_dict =  CustomDict(
            output_dict,
            error_message='Key not found. Indexing requires a tuple of Variable objects: derivatives[<of>, <wrt>]',
        )
    elif wrt_is_list:
        return_dict = CustomDict(
            {wrt: output_dict[ofs[0], wrt] for wrt in wrts},
            error_message='Key not found. Indexing requires a Variable object: derivatives[<wrt>]'
        )
    elif of_is_list:
        return_dict = CustomDict(
            {of: output_dict[of, wrts[0]] for of in ofs},
            error_message='Key not found. Indexing requires a Variable object: derivatives[<of>]'
        )
    else:
        return_dict = output_dict[ofs[0], wrts[0]]

    return return_dict


class CustomDict(dict):
    def __init__(self, *args, error_message = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_message = error_message

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as e:
            raise KeyError(f"{self.error_message}: {e}")

class TestDerivative(csdl_tests.CSDLTest):
    
    def test_docstring(self):
        self.docstest(derivative)

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

        dz_dx = csdl.derivative(ofs = z, wrts = x)
        dz_dx.add_name('dz_dx')
        print(dz_dx.value) 
        print((y+1.0).value) 

        dz2_dxy = csdl.derivative(ofs = dz_dx, wrts = y)
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

        dz_dx = csdl.derivative(ofs = z, wrts = x)
        dz_dx.add_name('dz_dx')
        # print(dz_dx.value) 
        # print(np.diagflat((y+1.0).value)) 

        dz2_dxy = csdl.derivative(ofs = dz_dx, wrts = y)
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
        print(csdl.derivative(ofs = s3, wrts = x).value)

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
        deriv = csdl.derivative(ofs = x, wrts = [b])[b]
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
        # csdl.derivative(ofs = s1, wrts = [x,y])
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

    # def issue(self,):
    #     # Define variables: using openmdao solved optimization values
    #     import csdl_alpha as csdl
    #     z1 = csdl.Variable(name = 'z1', value = 1.97763888)
    #     z2 = csdl.Variable(name = 'z2', value = 8.83056605e-15)
    #     x = csdl.Variable(name = 'x', value = 0.0)
    #     y2 = csdl.ImplicitVariable(name = 'y2', value = 1.0)

    #     # Define each "component" from the example
    #     with csdl.namespace('Discipline 1'):
    #         y1 = z1**2 + z2 + x - 0.2*y2
    #         y1.add_name('y1')

    #     with csdl.namespace('Discipline 2'):
    #         residual = csdl.sqrt(y1) + z1 + z2 - y2
    #         residual.add_name('residual')

    #     # Specifiy coupling
    #     with csdl.namespace('Couple'):
    #         solver = csdl.nonlinear_solvers.GaussSeidel()
    #         solver.add_state(y2, residual, state_update=y2+residual, tolerance=1e-8)
    #         solver.run()

    #     dy1dx = csdl.derivative(ofs = y1, wrt = x) # What should this value be???? Should this go through the nonlinear solver?? or stay inside the residual function???
        # dy1dx is not part of the residual function graph so it should be IFTed



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