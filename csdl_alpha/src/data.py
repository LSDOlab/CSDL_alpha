def inline_save(filename:str):
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
    for key in recorder.node_graph_map.keys():
        if isinstance(key, Variable):
            if key._save:
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
                dset = inline_grp.create_dataset(savename, data=key.value)
                # The shape is already stored in the value
                # dset.attrs['shape'] = key.shape
                if key.tags:
                    dset.attrs['tags'] = key.tags
                if key.hierarchy is not None:
                    dset.attrs['hierarchy'] = key.hierarchy
                if key.names:
                    dset.attrs['names'] = key.names
    f.close()


def import_h5py(filename:str, group:str):
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
    
def save_optimization_variables():
    """Save optimization variables.

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

def save_all_variables():
    """Save all variables in the current recorder's node graph."""
    from ..api import Variable
    from ..api import get_current_recorder

    recorder = get_current_recorder()
    for key in recorder.node_graph_map.keys():
        if isinstance(key, Variable):
            key.save()