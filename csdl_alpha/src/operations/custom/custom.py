from csdl_alpha.utils.parameters import Parameters
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.utils.inputs import variablize, get_type_string, ingest_value
from csdl_alpha.src.graph.node import Node
from csdl_alpha.utils.typing import VariableLike
import numpy as np

class CustomOperation(Operation):
    def __init__(self, *args, **kwargs):
        self.parameters = Parameters()
        self.parameters.hold(kwargs)
        self.initialize()
        self.parameters.check(kwargs)

        self.input_dict = {}
        self.output_dict = {}
        self.derivative_parameters = {}

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

class CustomExplicitOperation(CustomOperation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.evaluate = self._wrap_evaluate(self.evaluate)
        self.locked = False

    def initialize(self):
        """
        Declare parameters here.
        """
        pass

    def evaluate(self):
        raise NotImplementedError('not implemented')

    def compute(self, inputs, outputs):
        raise NotImplementedError('not implemented')

    def compute_derivatives(self, inputs, outputs, derivatives):
        raise NotImplementedError('not implemented')

    def _wrap_evaluate(self, evaluate):
        
        def new_evaluate(*args, **kwargs):

            # If the evaluate method is called multiple times, raise an error
            if self.locked:
                raise RuntimeError('Cannot call evaluate multiple times on the same CustomExplicitOperation object. Create a new object instead.')
            self.locked = True

            # Node.__init__(self)
            # self.name = self.__class__.__name__

            # self.recorder._add_node(self)
            eval_outputs = evaluate(*args, **kwargs)
            Operation.__init__(self, *list(self.input_dict.values()))

            self.set_outputs(list(self.output_dict.values()))

            self.finalize_and_return_outputs()

            return eval_outputs
        return new_evaluate

    def compute_inline(self, *args):
        inputs = {key:input for key, input in zip(self.input_dict.keys(), args)}
        comp_outputs = {}

        inputs = preprocess_custom_inputs(inputs)
        self.compute(inputs, comp_outputs)
        comp_outputs = postprocess_custom_outputs(comp_outputs, self.output_dict)

        output = [comp_outputs[key] for key in self.output_dict.keys()]
        if len(output) == 1:
            output = output[0]
        return output

    # def set_inline_values(self):
    #     inputs = {key: input.value for key, input in self.input_dict.items()}
    #     comp_outputs = {}

    #     self.compute(inputs, comp_outputs)

    #     for key, output in self.output_dict.items():
    #         output.set_value(comp_outputs[key])

    def create_output(self, name:str, shape:tuple):
        """Create and store an output variable. 

        This method creates a new output variable with the given name and shape,
        and populates the `outputs` dictionary of the object. 

        Parameters
        ----------
        name : str
            The name of the output variable.
        shape : tuple
            The shape of the output variable.

        Returns
        -------
        Variable
            The created output variable that represents the output of the operation.
        """
        output = Variable(shape)

        if name in self.output_dict.keys():
            raise KeyError(f'Output variable \'{name}\' already created.')

        self.output_dict[name] = output
        return output
    
    def declare_input(self, key:str, variable):
        """Declares a variable as an input.

        Defines the 'inputs' dictionary for the 'compute' method. 
        Sets the given input variable in the 'inputs' dictionary under the given key.

        Parameters
        ----------
        key : str
            The key for the variable.
        variable : csdl.Variable
            The input variable.
        """
        if key in self.input_dict.keys():
            raise KeyError(f'Input variable \'{key}\' already declared.')
        try:
            self.input_dict[key] = variablize(variable)
        except Exception as e:
            raise TypeError(f'Error with input variable \'{key}\': {e}')

    def declare_derivative_parameters(
        self,
        of, wrt,
        dependent=True,
        rows=None, cols=None,
        val=None,
        method='exact', step=None, form=None, step_calc=None,
    ):
        """Declare derivative parameters for computing derivatives.

        Parameters
        ----------
        of : str or list
            The variable(s) to take the derivative of.
        wrt : str or list
            The variable(s) to take the derivative with respect to.
        dependent : bool, optional
            Whether the derivative is dependent on other variables, by default True.
        rows : int, optional
            The number of rows in the derivative matrix, by default None.
        cols : int, optional
            The number of columns in the derivative matrix, by default None.
        val : float, optional
            The initial value of the derivative, by default None.
        method : str, optional
            The method used to compute the derivative, by default 'exact'.
        step : float, optional
            The step size used in numerical differentiation, by default None.
        form : str, optional
            The form of the derivative, by default None.
        step_calc : str, optional
            The step calculation method, by default None.

        Raises
        ------
        TypeError
            If `of` is not a string or list.
        TypeError
            If `wrt` is not a string or list.
        KeyError
            If the derivative for the given variables is already declared.
        """            
        # check argument types
        if not isinstance(of, (str, list)):
            raise TypeError(
                'of must be a string or list; {} given'.format(
                    type(of)))
        if not isinstance(wrt, (str, list)):
            raise TypeError(
                'wrt must be a string or list; {} given'.format(
                    type(wrt)))

        # user-provided lists of variables of wildcards
        of_list = []
        wrt_list = []
        if isinstance(of, str):
            if of == '*':
                of_list = list(self.output_dict.keys())
        elif isinstance(of, list):
            if any(x == '*' for x in of):
                of_list = list(self.output_dict.keys())
            else:
                of_list = of
        if isinstance(wrt, str):
            if wrt == '*':
                wrt_list = list(self.input_dict.keys())
        elif isinstance(wrt, list):
            if any(x == '*' for x in wrt):
                wrt_list = list(self.input_dict.keys())
            else:
                wrt_list = wrt

        # declare each derivative one by one
        if len(of_list) > 0 and len(wrt_list) > 0:
            for a in of_list:
                for b in wrt_list:
                    self.declare_derivative_parameters(
                        a,
                        b,
                        dependent=dependent,
                        rows=rows,
                        cols=cols,
                        val=val,
                        method=method,
                        step=step,
                        form=form,
                        step_calc=step_calc,
                    )
        elif len(of_list) > 0:
            for a in of_list:
                self.declare_derivative_parameters(
                    a,
                    wrt=wrt,
                    dependent=dependent,
                    rows=rows,
                    cols=cols,
                    val=val,
                    method=method,
                    step=step,
                    form=form,
                    step_calc=step_calc,
                )
        elif len(wrt_list) > 0:
            for b in wrt_list:
                self.declare_derivative_parameters(
                    of,
                    b,
                    dependent=dependent,
                    rows=rows,
                    cols=cols,
                    val=val,
                    method=method,
                    step=step,
                    form=form,
                    step_calc=step_calc,
                )
        else:
            if (of, wrt) in self.derivative_parameters.keys():
                raise KeyError(
                    'Derivative {} wrt {} already declared'.format(
                        of, wrt))
            self.derivative_parameters[of, wrt] = dict()
            self.derivative_parameters[of, wrt]['dependent'] = dependent
            self.derivative_parameters[of, wrt]['rows'] = rows
            self.derivative_parameters[of, wrt]['cols'] = cols
            self.derivative_parameters[of, wrt]['val'] = val
            self.derivative_parameters[of, wrt]['method'] = method
            self.derivative_parameters[of, wrt]['step'] = step
            self.derivative_parameters[of, wrt]['form'] = form
            self.derivative_parameters[of, wrt]['step_calc'] = step_calc
    
    def evaluate_diagonal_jacobian(self, *args):
        raise NotImplementedError('not implemented') 

    def evaluate_jvp(self, *args):
        raise NotImplementedError('not implemented')

    def evaluate_vjp(self, *args):
        raise NotImplementedError('not implemented')

