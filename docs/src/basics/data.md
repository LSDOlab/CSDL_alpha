# Data

This page gives an overview of saving and loading data in CSDL. The standard data format for CSDL is `.hdf5`

## Saving Variables

Saving CSDL variables is a two step process. The first step is to indicate what variables should be saved. This is done by calling `.save()` on the variable. This indicates to the backend that a variable should be saved. Additional methods are provided to set groups of variables to be saved, such as the `.save()` method on the `VariableGroup` class, and the `save_optimization_variables()` and `save_all_variables()` functions.

The second step is to actually save the variables. This is generally done by the backend. When using inline mode, you can save the variables by calling the `inline_save()` function. This will create an `.hdf5` file with the saved variables, including their value, names, and tags.

## Loading Variables

Loading variables is done by calling the `import_h5py` function on the `Recorder` object. This will load the variables from the `.hdf5` file, and assign the values to the variables, returning a dictionary of variables keyed by their names. You can then access the variables by their names, and use them in your model.