from typing import Union
from csdl_alpha.utils.inputs import listify_variables

def inline_export(filename:str, summary_csv:bool=False, do_print=False):
    """Save variables from the current recorder's node graph to an HDF5 file.

    Parameters
    ----------
    filename : str
        The name of the HDF5 file to save the variables to.
    summary_csv : bool, optional
        If True, a CSV file will be saved instead of an HDF5 file, by default False.
    do_print : bool, optional
        If True, the CSV file will be printed to the console, by default False.
    """
    if summary_csv:
        _inline_csv_save(filename, print_csv=do_print)
    else:
        _export_h5py(filename, do_print=do_print)

def _export_h5py(filename:str, do_print=False):
    """Save variables from the current recorder's node graph to an HDF5 file.

    Parameters
    ----------
    filename : str
        The name of the HDF5 file to save the variables to.
    """
    import h5py
    from ..src.graph.variable import Variable
    from ..api import get_current_recorder
    
    if not filename.endswith('.hdf5'):
        filename = f'{filename}.hdf5'
    f = h5py.File(filename, 'w')

    inline_grp = f.create_group('inline')
    recorder = get_current_recorder()
    name_counter_dict = {}
    for key, index in recorder.active_graph.node_table.items():
        if isinstance(key, Variable):
            if key._save:
                savename = _get_savename(key, name_counter_dict)
                try:
                    dset = inline_grp.create_dataset(savename, data=key.value)
                except ValueError:
                    # probably has the same name as another, so add the index
                    savename = f'{savename}_{index}'
                    dset = inline_grp.create_dataset(savename, data=key.value)
                # The shape is already stored in the value
                dset.attrs['index'] = index
                if key.tags:
                    dset.attrs['tags'] = key.tags
                if key.hierarchy is not None:
                    dset.attrs['hierarchy'] = key.hierarchy
                if key.names:
                    dset.attrs['names'] = key.names
    f.close()

def _get_savename(key, name_counter_dict):
    if not key.names:
        if not key.namespace.prepend in name_counter_dict:
            name_counter_dict[key.namespace.prepend] = 0
        name_count = name_counter_dict[key.namespace.prepend]
        name_counter_dict[key.namespace.prepend] += 1
        if key.namespace.prepend is None:
            savename = f'variable_{name_count}'
        else:
            savename = f'{key.namespace.prepend}.variable_{name_count}'
    else:
        savename = key.names[0]
    return savename

def inline_import(filename:str, group:str):
    """
    Import variables from an HDF5 file.

    Parameters
    ----------
    filename : str
        The path to the HDF5 file.
    group : str
        The name of the group within the HDF5 file (eg, 'inline', 'iteration1').

    Returns
    -------
    dict
        A dictionary containing the imported variables, where the keys are the variable names and the values are instances of the Variable class.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    KeyError
        If the specified group does not exist within the file.
    """
    import h5py
    from ..src.graph.variable import Variable

    f = h5py.File(f'{filename}', 'r')
    grp = f[group]

    # Iterate over the keys in the group
    variables = {}
    for key in grp.keys():
        # Get the dataset
        dataset = grp[key]

        # Create a Variable instance with the dataset value and name
        variable = Variable(value=dataset[...], name=key)

        # add attributes to the variable
        if 'tags' in dataset.attrs:
            variable.tags = dataset.attrs['tags']
        if 'hierarchy' in dataset.attrs:
            variable.hierarchy = dataset.attrs['hierarchy']
        if 'names' in dataset.attrs:
            variable.names = dataset.attrs['names']
        variables[key] = variable

    f.close()

    # Return the dictionary of variables
    return variables
    
def _inline_csv_save(filename:str, print_csv:bool=False):
    """Save the name, min, max, and mean of variables from the current recorder's node graph to a CSV file.

    Parameters
    ----------
    filename : str
        The name of the CSV file to save the variables to.
    """
    import csv
    from ..api import get_current_recorder
    from ..src.graph.variable import Variable
    import numpy as np

    if not filename.endswith('.csv'):
        filename = f'{filename}.csv'

    recorder = get_current_recorder()
    insights = recorder.gather_insights()
    name_counter_dict = {}
    max_var_len = 0
    with open(filename, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Variable', 'Min', 'Max', 'Mean', 'Shape', 'Graphs'])
        for key in insights['nodes2graphs']:
            if isinstance(key, Variable):
                if key._save:
                    savename = _get_savename(key, name_counter_dict)
                    if len(savename) > max_var_len:
                        max_var_len = len(savename)

                    # in_graphs = ''
                    # if key in insights['nodes2graphs']:
                    in_graphs = ','.join([graph.name for graph in insights['nodes2graphs'][key]])

                    if key.value is not None:
                        value = key.value
                        csv_writer.writerow([savename, np.min(value), np.max(value), np.mean(value), value.shape, in_graphs])
                    else:
                        csv_writer.writerow([savename, None, None, None, None, in_graphs])

    if print_csv:
        name_len = max_var_len+5
        with open(f"{filename}", 'r') as f:
            csv_f = csv.reader(f)
            with np.printoptions(precision=3, suppress=True):

                for row in csv_f:
                    print('{:<{}}  {:<30}  {:<30} {:<30} {:<10} {:<30}'.format(row[0], name_len, row[1], row[2], row[3], row[4], row[5]))
                
def save_optimization_variables():
    """Sets optimization variables to be saved.

    This function sets the optimization variables to be saved, including objectives, constraints, and design variables.
    """
    from ..api import get_current_recorder

    recorder = get_current_recorder()
    for key in recorder.objectives:
        key.save()
    for key in recorder.constraints:
        key.save()
    for key in recorder.design_variables:
        key.save()

def save_all_variables(ignore_unnamed = False):
    """Sets all variables in the current recorder's node graph to be saved.
    
    Parameters
    ----------
    ignore_unnamed : bool, optional
        If True, only variables with names will be saved, by default False.
    """
    from ..api import Variable
    from ..api import get_current_recorder

    recorder = get_current_recorder()
    for key in recorder.node_graph_map.keys():
        if isinstance(key, Variable):
            if ignore_unnamed:
                if key.name:
                    key.save()
            else:
                key.save()


def save_h5py_variable(inline_grp, variable, savename:str):
    try:
        dset = inline_grp.create_dataset(savename, data=variable.value)
    except ValueError:
        # probably has the same name as another, so add the index
        savename = f'{savename}_{variable.index}'
        dset = inline_grp.create_dataset(savename, data=variable.value)
    # The shape is already stored in the value
    if variable.tags:
        dset.attrs['tags'] = variable.tags
    if variable.hierarchy is not None:
        dset.attrs['hierarchy'] = variable.hierarchy
    if variable.names:
        dset.attrs['names'] = variable.names

def save_h5py_variables(
        filename:str,
        groupname:str,
        variables:Union['Variable',list['Variable']],
    ):
    import h5py
    variables = listify_variables(variables)
    if not isinstance(groupname, str):
        groupname = str(groupname)

    if not filename.endswith('.hdf5'):
        filename = f'{filename}.hdf5'
    f = h5py.File(filename, 'a')
    inline_grp = f.create_group(groupname)

    name_counter_dict = {}
    for variable in variables:
        savename = _get_savename(variable, name_counter_dict)
        save_h5py_variable(inline_grp, variable, savename)

    f.close()
