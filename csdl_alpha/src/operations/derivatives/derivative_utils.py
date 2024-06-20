from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.graph import Graph, is_variable
from csdl_alpha.utils.printing import print_tabularized

import numpy as np
from typing import Union

def verify_derivatives(
        ofs:Union[Variable, list[Variable], tuple[Variable]],
        wrts:Union[Variable, list[Variable], tuple[Variable]],
        step_size:float,
        print_results:bool = True,
        raise_on_error:bool = True,
        verification_options:dict[tuple[Variable,Variable],dict] = None,
        derivative_kwargs:dict = None,
        backend = 'inline',
        )->None:
    # Check arguments:
    ofs, wrts = listify_ofs_and_wrts(ofs, wrts)
    if backend not in ['inline', 'jax']:
        raise ValueError(f"Backend {backend} not supported")
    if backend == 'inline':
        forward_evaluation = None
    else:
        from csdl_alpha.backends.jax.graph_to_jax import create_jax_interface
        forward_evaluation = create_jax_interface(wrts, ofs)

    if derivative_kwargs is None:
        derivative_kwargs = {}
    elif not isinstance(derivative_kwargs, dict):
        raise TypeError(f"Derivative kwargs must be a dictionary. Type {type(derivative_kwargs)} given")

    # finite difference values to check against
    import csdl_alpha as csdl
    import time as time
    finite_differenced_values = {}
    graph = csdl.get_current_recorder().active_graph
    start = time.time()
    finite_differenced_values = csdl.derivative_utils.finite_difference(ofs, wrts, step_size=step_size, forward_evaluation=forward_evaluation)
    end = time.time()
    print(f"Finite difference took {end - start} seconds")

    # analytical derivatives to validate
    analytical_derivative_values = {}
    start = time.time()
    deriv = csdl.derivative(ofs = ofs, wrts=wrts,**derivative_kwargs)
    for of_ind, of in enumerate(ofs):
        for wrt_ind, wrt in enumerate(wrts):
            analytical_derivative_values[(of, wrt)] = {}
            analytical_derivative_values[(of, wrt)]['value'] = deriv[of, wrt].value
            analytical_derivative_values[(of, wrt)]['fd_value'] = finite_differenced_values[of,wrt]

            if verification_options is not None:
                if (of, wrt) in verification_options:
                    analytical_derivative_values[(of, wrt)]['max_rel_error'] = verification_options[of, wrt]['max_rel_error']
                    analytical_derivative_values[(of, wrt)]['tag'] = verification_options[of, wrt]['tag']

            if wrt.name is None:
                wrt_name = f'wrt {wrt_ind}'
            else:
                wrt_name = wrt.name

            if of.name is None:
                of_name = f'of {of_ind}'
            else:
                of_name = of.name

            analytical_derivative_values[(of, wrt)]['of_name'] = of_name
            analytical_derivative_values[(of, wrt)]['wrt_name'] = wrt_name

    if backend == 'inline':
        graph.execute_inline()
        for of_ind, of in enumerate(ofs):
            for wrt_ind, wrt in enumerate(wrts):
                analytical_derivative_values[(of, wrt)]['value'] = deriv[of, wrt].value
    else:
        from csdl_alpha.backends.jax.graph_to_jax import create_jax_interface
        deriv_evaluation = create_jax_interface(wrts, [deriv[key] for key in deriv])
        jax_derivs = deriv_evaluation({wrt:wrt.value for wrt in wrts})
        for of_ind, of in enumerate(ofs):
            for wrt_ind, wrt in enumerate(wrts):
                analytical_derivative_values[(of, wrt)]['value'] = jax_derivs[deriv[of, wrt]]

    end = time.time()
    print(f"Analytical took {end - start} seconds")

    return verify_derivative_values(
        derivative_values = analytical_derivative_values,
        print_results = print_results,
        raise_on_error = raise_on_error,
    )

def verify_derivative_values(
        derivative_values: dict[tuple[Variable,Variable], dict[str, np.array]],
        print_results:bool = True,
        raise_on_error:bool = True,
    )->dict[tuple[Variable,Variable], dict[str, np.array]]:
    """Given analytical derivatives, this function verifies that the analytical
    derivatives are correct using finite differences.

    Parameters
    ----------
    derivative_values : dict[tuple[Variable,Variable], np.array]
        dictionary of the analytical derivatives to verify.

        keys of 'derivative_values[<of variable>, <wrt variable>]':
        required keys:
            - 'value': np.array of the analytical derivative value
            - 'fd_value': np.array of the finite difference derivative value
        options:
            - 'of_name': str of the output variable to print
            - 'wrt_name': str of the input variable to print
            - 'tag': str to print in the results table
            - 'max_rel_error': float of the maximum relative error allowed
    print_results : bool, optional
       If true, prints a table of derivative errors, by default True
    raise_on_error : bool, optional
       If true, raises an error if the relative error is greater than the maximum relative error, by default True
    """
    all_ofs = set()
    all_wrts = set()
    # check derivative dictionary
    if not isinstance(derivative_values, dict):
        raise TypeError(f"Derivative values must be a dictionary. Type {type(derivative_values)} given")
    for key in derivative_values:
        if not isinstance(key, tuple):
            raise TypeError(f"Keys of derivative values must be tuples. Type {type(key)} given")
        if not isinstance(derivative_values[key], dict):
            raise TypeError(f"Values of derivative values must be dictionaries. Type {type(derivative_values[key])} given")
        if 'value' not in derivative_values[key]:
            raise KeyError(f"Derivative values must have a 'value' key")
        else:
            if not isinstance(derivative_values[key]['value'], np.ndarray):
                raise TypeError(f"Derivative values must be numpy arrays. Type {type(derivative_values[key]['value'])} given")
        if 'fd_value' not in derivative_values[key]:
            raise KeyError(f"Derivative values must have a 'fd_value' key")
        else:
            if not isinstance(derivative_values[key]['fd_value'], np.ndarray):
                raise TypeError(f"Derivative values must be numpy arrays. Type {type(derivative_values[key]['fd_value'])} given")
        
        if 'max_rel_error' not in derivative_values[key]:
            derivative_values[key]['max_rel_error'] = 1e-5
        if 'tag' not in derivative_values[key]:
            derivative_values[key]['tag'] = ''
        if 'of_name' not in derivative_values[key]:
            derivative_values[key]['of_name'] = str(key[0])
        if 'wrt_name' not in derivative_values[key]:
            derivative_values[key]['wrt_name'] = str(key[1])

        all_ofs.add(key[0])
        all_wrts.add(key[1])
    num_ofs = len(all_ofs)
    num_wrts = len(all_wrts)

    # Print the results
    table_rows = []
    # Create a table of the results, each row is a different derivative
    for key in derivative_values:

        of_name = derivative_values[key]['of_name']
        wrt_name = derivative_values[key]['wrt_name']

        analytical_jac = derivative_values[key]['value']
        fd_jac = derivative_values[key]['fd_value']

        norm = np.linalg.norm(analytical_jac)
        fd_norm = np.linalg.norm(fd_jac)

        error = np.linalg.norm(analytical_jac - fd_jac)
        derivative_values[key]['abs_error'] = error

        max_rel_error = derivative_values[key]['max_rel_error']
        pair_tag = derivative_values[key]['tag']

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
                            table_keys=[f'ofs ({num_ofs})', f'wrts ({num_wrts})', 'norm', 'fd norm', 'error', 'rel error', 'tags'],
                            table_rows=table_rows,
                            title = 'Derivative Verification Results'
                        )
                        raise ValueError(f"High error in derivative verification: {rel_error}")

    if print_results:
        print_tabularized(
            table_keys=[f'ofs ({num_ofs})', f'wrts ({num_wrts})', 'norm', 'fd norm', 'error', 'rel error', 'tags'],
            table_rows=table_rows,
            title = 'Derivative Verification Results'
        )
    return derivative_values

def finite_difference(
        ofs: list[Variable],
        wrts: list[Variable],
        step_size: float,
        forward_evaluation: callable = None,
    )->dict[tuple[Variable,Variable], np.array]:
    """WARNING: This function is not differentiable should only be used for debugging purposes. 
    Computes the derivatives of the output variables with respect to an input variable using finite difference: (f(x + h) - f(x)) / h.

    Parameters
    ----------
    ofs : list[Variable]
        A list of output variables to compute the derivative of using finite differences.
    wrts : list[Variable]
        A list of input variables to perturb and compute the derivatives with respect to.
    step_size : float
        The step size to use for the finite difference.
    forward_evaluation : callable, optional
        A function that takes in the input variables and computes the output variables.
        If not provided, the current graph will be used using inline calculation.
        The forward evaluation function must have the format: outs:dict[Variable, np.array] = forward_evaluation(wrt:dict[Variable, np.array])
        where the output and input variable must match the variables in ofs and wrts respectively.
    
    Returns
    -------
    dict[tuple[Variable,Variable], np.array]
        A dictionary of the derivative values of the output variables with respect to the input variables.
    """
    import csdl_alpha as csdl
    # preprocessing:
    ofs, wrts = listify_ofs_and_wrts(ofs, wrts)
    if forward_evaluation is None:
        def forward_evaluation(wrts_arg:dict[Variable, np.array])->dict[Variable, np.array]:
            graph = csdl.get_current_recorder().active_graph
            for wrt in wrts_arg:
                wrt.value = wrts_arg[wrt]
            graph.execute_inline()
            return {of:of.value for of in ofs}

    # Store orginal values
    original_wrts = {wrt:wrt.value.copy() for wrt in wrts}
    new_wrts = {wrt:wrt.value.copy() for wrt in wrts}
    for original_wrt, orignal_val in original_wrts.items():
        if not isinstance(orignal_val, np.ndarray):
            raise ValueError(f"Variable {original_wrt} does not have a value: {orignal_val}")

    # Run the forward evaluation once
    outputs = forward_evaluation(original_wrts)
    original_outputs = {of:outputs[of].copy() for of in ofs}

    # Store the finite differenced values
    finite_differenced_dict = {}
    for of in ofs:
        for wrt in wrts:
            finite_differenced_dict[of, wrt] = np.zeros((of.size, wrt.size))

    # Perform the finite difference bt perturbing the input variable column by column
    for wrt in wrts:
        original_wrt_value = original_wrts[wrt]
        for col_index in range(wrt.size):
            # perturb the input variable
            perturbed_input = original_wrt_value.flatten()
            perturbed_input[col_index] += step_size
            new_wrts[wrt] = perturbed_input.reshape(wrt.shape)
            perturbed_outputs = forward_evaluation(new_wrts)

            # compute the output values
            for of in ofs:
                fx_plus_h = perturbed_outputs[of].flatten()
                fx = original_outputs[of].flatten()
                h = step_size
                finite_differenced_dict[of, wrt][:, col_index] = (fx_plus_h - fx) / h

            # Reset the input variable
            new_wrts[wrt] = original_wrt_value

    # Reset all values
    # TODO: Shouldn't be necessary for non-inline
    _ = forward_evaluation(original_wrts)

    return finite_differenced_dict

def listify_ofs_and_wrts(ofs, wrts):
    if not isinstance(ofs, (list, tuple)):
        ofs = [ofs]
    if not isinstance(wrts, (list, tuple)):
        wrts = [wrts]

    for of in ofs:
        if not is_variable(of):
            raise ValueError(f"{of} is not a variable")
    for wrt in wrts:
        if not is_variable(wrt):
            raise ValueError(f"{wrt} is not a variable")
    return ofs, wrts