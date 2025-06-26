import numpy as np
import scipy.sparse as sp
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import variablize, get_type_string, ingest_value

def process_custom_derivatives_metadata(
        derivative_dict:dict[tuple[str,str], dict],
        out_dict:dict[str, Variable],
        in_dict:dict[str, Variable],
    ):
    """
    TODO: Add tests

    processes derivative metadata.
    given derivative metadata dict, processes:
    - standard dense numpy
    - sparse rows and columns given
    - sparse rows and columns and vals given
    - derivative not declared (zeros)
    """

    for derivative_tuple in derivative_dict:
        given_rows = derivative_dict[derivative_tuple]['rows']
        given_cols = derivative_dict[derivative_tuple]['cols']
        given_val = derivative_dict[derivative_tuple]['val']

        size_out = np.prod(out_dict[derivative_tuple[0]].shape)
        size_in = np.prod(in_dict[derivative_tuple[1]].shape)

        derivative_dict[derivative_tuple]['size_out'] = size_out
        derivative_dict[derivative_tuple]['size_in'] = size_in

        if given_rows is not None and given_cols is not None:
            if given_val is None:
                derivative_dict[derivative_tuple]['backend_type'] = 'row_col_given'
            elif given_val is not None:
                derivative_dict[derivative_tuple]['backend_type'] = 'row_col_val_given'
                derivative_dict[derivative_tuple]['given_val'] = sp.csc_matrix((given_val, (given_rows, given_cols)), shape=(size_out, size_in))
        elif given_val is not None:
            derivative_dict[derivative_tuple]['backend_type'] = 'row_col_val_given'

            if isinstance(given_val, np.ndarray):
                derivative_dict[derivative_tuple]['given_val'] = given_val.reshape((size_out, size_in))
            elif sp.issparse:
                if given_val.shape != (size_out, size_in):
                    raise ValueError(f'sparse partials {derivative_tuple} is of incorrect shape. {given_val.shape} != {(size_out, size_in)}')
                derivative_dict[derivative_tuple]['given_val'] = given_val
            else:
                derivative_dict[derivative_tuple]['given_val'] = given_val*np.ones((size_out, size_in))
        elif derivative_dict[derivative_tuple]['dependent'] is False:
            derivative_dict[derivative_tuple]['backend_type'] = 'row_col_val_given'
            derivative_dict[derivative_tuple]['given_val'] = sp.csc_matrix((size_out, size_in))

        elif (given_rows is None) and (given_cols is None) and (given_val is None):
            derivative_dict[derivative_tuple]['backend_type'] = 'standard'
        else:
            raise ValueError(f'declare derivative arguments for {derivative_tuple} is incorrect.')

    for out_str in out_dict:
        for in_str in in_dict:
            derivative_tuple = (out_str, in_str)
            if derivative_tuple not in derivative_dict:
                size_out = np.prod(out_dict[derivative_tuple[0]].shape)
                size_in = np.prod(in_dict[derivative_tuple[1]].shape)

                derivative_dict[derivative_tuple] = {}
                derivative_dict[derivative_tuple]['size_out'] = size_out
                derivative_dict[derivative_tuple]['size_in'] = size_in
                derivative_dict[derivative_tuple]['backend_type'] = 'standard'
                derivative_dict[derivative_tuple]['sparse'] = False

def prepare_compute_derivatives(derivative_meta:dict[tuple[str,str], dict])->dict:

    pre_allocated_derivatives = {}

    # Set derivatives
    for derivative_tuple in derivative_meta:

        # If rows and cols are given, give a flat vector with size len(rows) or size len(cols)
        if derivative_meta[derivative_tuple]['backend_type'] == 'row_col_given':
            len_val = len(derivative_meta[derivative_tuple]['rows'])
            pre_allocated_derivatives[derivative_tuple] = np.zeros((len_val, ))
        elif derivative_meta[derivative_tuple]['backend_type'] == 'row_col_val_given':
            pass
        else:
            # Otherwise, give zeros of 2D jac matrix
            size_out = derivative_meta[derivative_tuple]['size_out']
            size_in = derivative_meta[derivative_tuple]['size_in']

            if derivative_meta[derivative_tuple]['sparse'] is True:
                pre_allocated_derivatives[derivative_tuple] = sp.csc_matrix((size_out, size_in))
            else:
                pre_allocated_derivatives[derivative_tuple] = np.zeros((size_out, size_in))

    return pre_allocated_derivatives

def postprocess_compute_derivatives(
        totals:dict[tuple[str,str], Variable],
        derivative_meta:dict[tuple[str,str], dict],
    ):

    # Post-process user given derivatives
    for derivative_tuple in derivative_meta:
        size_out = derivative_meta[derivative_tuple]['size_out']
        size_in = derivative_meta[derivative_tuple]['size_in']

        if derivative_meta[derivative_tuple]['backend_type'] == 'row_col_val_given':
            # If the value is given in define, use that.
            totals[derivative_tuple] = derivative_meta[derivative_tuple]['given_val']
        elif derivative_meta[derivative_tuple]['backend_type'] == 'row_col_given':

            # If the rows and cols are given, create sparse matrix of only vals.
            given_rows = derivative_meta[derivative_tuple]['rows']
            given_cols = derivative_meta[derivative_tuple]['cols']
            totals[derivative_tuple] = sp.csc_matrix((totals[derivative_tuple], (given_rows, given_cols)), shape=(size_out, size_in))
        else:
            # If standard derivative, just use user-given derivatie directly.
            totals[derivative_tuple] = totals[derivative_tuple].reshape((size_out, size_in))

    for total_tuple in totals:
        if total_tuple not in derivative_meta:
            raise KeyError(f'derivative {total_tuple} does not exist')
        

# https://stackoverflow.com/questions/19022868/how-to-make-dictionary-read-only
def _readonly(self, *args, **kwargs):
    raise RuntimeError("Cannot modify inputs dictionary.")

# https://stackoverflow.com/questions/19022868/how-to-make-dictionary-read-only
class CustomInputsDict(dict):
    __setitem__ = _readonly
    __delitem__ = _readonly
    pop = _readonly
    popitem = _readonly
    clear = _readonly
    update = _readonly
    setdefault = _readonly

def preprocess_custom_inputs(inputs):
    return CustomInputsDict(inputs)

def postprocess_custom_outputs(given_outputs:dict, declared_outputs:dict):
    processed_outputs = {}
    for given_key, given_output in given_outputs.items():

        # If they give an output that isn't a VariableLike, raise an error
        try:
            given_output = ingest_value(given_output)
        except Exception as e:
            raise TypeError(f'Error with output \'{given_key}\': {e}')

        # If they give an output that wasn't declared, raise an error
        if given_key not in declared_outputs:
            raise KeyError(f'Output \'{given_key}\' was not declared but was computed')
        
        # If they give an output that doesn't have the right shape, raise an error
        if given_output.size == 1: # broadcasting????
            given_output = np.ones(declared_outputs[given_key].shape) * given_output.flatten()
        elif given_output.shape != declared_outputs[given_key].shape:
            raise ValueError(f'Output \'{given_key}\' must have shape {declared_outputs[given_key].shape}, but shape {given_output.shape} was given')

        processed_outputs[given_key] = given_output

    for declared_key, declared_output_variable in declared_outputs.items():

        # If they didn't give an output that was declared, raise an error
        if declared_key not in processed_outputs:
            raise KeyError(f'Output \'{declared_key}\' was declared but was not computed')

    return processed_outputs

def prepare_d_inputs(inputs_meta):
    d_inputs = {}
    for input_var, input_str in inputs_meta.items():
        d_inputs[input_str] = np.zeros(input_var.shape)
    return d_inputs

def is_overwritten(child_class, parent_class, method_name):
    """
    Check if a method is overwritten in a child class compared to a parent class.

    Parameters:
    child_class (type): The child class to check.
    parent_class (type): The parent class to compare against.
    method_name (str): The name of the method to check.

    Returns:
    bool: True if the method is overwritten in the child class, False otherwise.
    """
    child_method = getattr(child_class, method_name, None)
    parent_method = getattr(parent_class, method_name, None)
    return child_method != parent_method