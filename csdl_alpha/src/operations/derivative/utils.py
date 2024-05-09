from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.graph import Graph, is_variable
from csdl_alpha.utils.printing import print_tabularized

import numpy as np

def verify_derivatives_inline(
        ofs:list[Variable],
        wrts:list[Variable],
        step_size:float,
        of_wrt_meta_data:dict = None,
        print_results:bool = True,
        raise_on_error:bool = True,
    )->None:
    """This function verifies that the analytical derivatives are correct.

    Parameters
    ----------
    ofs : Union[Variable, list[Variable]]
        The output variables to compute the derivative of.

    wrts : Union[Variable, list[Variable]]
        The input variables to compute the derivatives with respect to.    
    """
    # TODO: Clean up this function

    # Inputs and outputs must be variables
    wrt_name_dict = {}
    of_name_dict = {}
    for i,wrt in enumerate(wrts):
        if not is_variable(wrt):
            raise ValueError(f"{wrt} is not a variable")
        if wrt.name is None:
            wrt_name_dict[wrt] = f'wrt {i}'
        else:
            wrt_name_dict[wrt] = wrt.name
    for i,of in enumerate(ofs):
        if not is_variable(of):
            raise ValueError(f"{of} is not a variable")
        if of.name is None:
            of_name_dict[of] = f'of {i}'
        else:
            of_name_dict[of] = of.name

    import csdl_alpha as csdl
    # Verifies that the derivatives are correct using finite differences



    # finite difference values to check against
    import time as time
    finite_differenced_values = {}
    graph = csdl.get_current_recorder().active_graph
    start = time.time()
    for wrt in wrts:
        finited_differenced_values_wrts = finite_difference_inline(ofs, wrt, graph, tolerance=step_size)
        for of in ofs:
            finite_differenced_values[(of, wrt)] = finited_differenced_values_wrts[of]['jacobian']
    end = time.time()
    print(f"Finite difference took {end - start} seconds")

    # analytical derivatives to validate
    analytical_derivative_values = {}
    start = time.time()
    for of in ofs:
        deriv = csdl.derivative.reverse(of, wrts=wrts)
        for wrt in wrts:
            analytical_derivative_values[(of, wrt)] = deriv[wrt].value
            # print(f"Analytical derivative of {of.name} with respect to {wrt.name}: \n{deriv.value}\n")
    graph.execute_inline()
    end = time.time()
    print(f"Analytical took {end - start} seconds")

    # Print the results
    table_rows = []
    # Create a table of the results, each row is a different derivative
    for key in analytical_derivative_values:

        of_name = of_name_dict[key[0]]
        wrt_name = wrt_name_dict[key[1]]

        analytical_jac = analytical_derivative_values[key]
        fd_jac = finite_differenced_values[key]

        norm = np.linalg.norm(analytical_jac)
        fd_norm = np.linalg.norm(fd_jac)

        error = np.linalg.norm(analytical_jac - fd_jac)
        
        if of_wrt_meta_data is None:
            max_rel_error = 1e-5
            pair_tag = ''
        else:
            max_rel_error = of_wrt_meta_data[key[0], key[1]]['max_rel_error']
            pair_tag = of_wrt_meta_data[key[0], key[1]]['tag']


        tags = ''
        if abs(fd_norm) < 1e-15:
            rel_error = ''
        else:
            rel_error = error / fd_norm
            if rel_error > 1e-5:
                if error > 1e-4:
                    tags += 'high error, '
        tags += pair_tag
                
        table_row = [of_name, wrt_name, str(norm), str(fd_norm), str(error), str(rel_error), tags]
        table_rows.append(table_row)

        if raise_on_error:
            if not isinstance(rel_error, str):
                if rel_error > max(max_rel_error, 1e-5):
                    if error > 1e-4:
                        print('Analytical Jacobian: ')
                        print(analytical_jac)
                        print('Finite Difference Jacobian: ')
                        print(fd_jac)
                        print_tabularized(
                            table_keys=[f'ofs ({len(ofs)})', f'wrts ({len(wrts)})', 'norm', 'fd norm', 'error', 'rel error', 'tags'],
                            table_rows=table_rows,
                            title = 'Derivative Verification Results'
                        )
                        raise ValueError(f"High error in derivative verification: {rel_error}")

    if print_results:
        print_tabularized(
            table_keys=[f'ofs ({len(ofs)})', f'wrts ({len(wrts)})', 'norm', 'fd norm', 'error', 'rel error', 'tags'],
            table_rows=table_rows,
            title = 'Derivative Verification Results'
        )


def finite_difference_inline(
        ofs:list[Variable],
        wrt:Variable,
        graph:Graph,
        tolerance:float=1e-6,
    ):
    """Computes the derivatives of the output variables with respect to an input variable using finite differences: (f(x + h) - f(x)) / h.

    Parameters
    ----------
    ofs : list[Variable]
        _description_
    wrt : Variable
        _description_
    graph : Graph
        _description_
    """
    # Store original values for the outputs and input
    finite_differenced_values = {}
    for of in ofs:
        finite_differenced_values[of] = {}
        finite_differenced_values[of]['original_value'] = of.value.copy()
        finite_differenced_values[of]['jacobian'] = np.zeros((of.size, wrt.size))

    original_wrt_value = wrt.value

    # Perform the finite difference bt perturbing the input variable column by column
    for col_index in range(wrt.size):
        # perturb the input variable
        perturbed_input = original_wrt_value.flatten()
        perturbed_input[col_index] += tolerance
        wrt.value = perturbed_input.reshape(wrt.shape)
        graph.execute_inline()

        # compute the output values
        for of in ofs:
            fx_plus_h = of.value.flatten()
            fx = finite_differenced_values[of]['original_value'].flatten()
            h = tolerance
            finite_differenced_values[of]['jacobian'][:, col_index] = (fx_plus_h - fx) / h

        # Reset the input variable
        wrt.value = original_wrt_value
    
    # Reset all values
    graph.execute_inline()
    return finite_differenced_values

def get_uncontract_action(
        expand_to_shape:tuple[int],
        contraction_axes:tuple[int],
        )->str:
    """
    For vector-jacobianing, expand the given vjp back to an input shape.
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    action = ''
    for i in range(len(expand_to_shape)):
        if i in contraction_axes:
            continue
        action += alphabet[i]
    action += '->'
    for i in range(len(expand_to_shape)):
        action += alphabet[i]
    return action