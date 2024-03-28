def inline_save(filename:str):
    import h5py
    from ..src.graph.variable import Variable
    from ..api import get_current_recorder
    
    if not filename.endswith('.hdf5'):
        filename = f'{filename}.hdf5'
    f = h5py.File(filename, 'w')

    inline_grp = f.create_group('inline')
    recorder = get_current_recorder()
    name_counter = 0 # TODO: make the name counter different for each namespace
    for key in recorder.node_graph_map.keys():
        if isinstance(key, Variable):
            if key.save:
                if not key.names:
                    savename = f'{key.namespace.prepend}.variable_{name_counter}'
                    name_counter += 1
                else:
                    savename = key.names[0]
                dset = inline_grp.create_dataset(savename, data=key.value)
                # dset.attrs['shape'] = key.shape
                if key.tags:
                    dset.attrs['tags'] = key.tags
                if key.hierarchy is not None:
                    dset.attrs['hierarchy'] = key.hierarchy
                if key.names:
                    dset.attrs['names'] = key.names
    f.close()


def import_h5py(filename:str, group:str):
    import h5py
    from ..src.graph.variable import Variable
    f = h5py.File(f'{filename}', 'r')
    grp = f[group]
    variables = {}
    for key in grp.keys():
        dataset = grp[key]
        variable = Variable(value=dataset[...], name=key)
        if 'tags' in dataset.attrs:
            variable.tags = dataset.attrs['tags']
        if 'hierarchy' in dataset.attrs:
            variable.hierarchy = dataset.attrs['hierarchy']
        if 'names' in dataset.attrs:
            variable.names = dataset.attrs['names']
        variables[key] = variable
    f.close()
    return variables
    
def save_optimization_variables():
    from ..api import get_current_recorder

    recorder = get_current_recorder()
    for key in recorder.objectives:
        key.save = True
    for key in recorder.constraints:
        key.save = True
    for key in recorder.design_variables:
        key.save = True

def save_all_variables():
    from ..api import get_current_recorder

    recorder = get_current_recorder()
    for key in recorder.node_graph_map.keys():
        key.save = True