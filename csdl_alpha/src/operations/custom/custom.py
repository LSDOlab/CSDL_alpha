from csdl_alpha.utils.parameters import Parameters
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.utils.inputs import variablize, get_type_string, ingest_value
from csdl_alpha.src.operations.custom.utils import (
    prepare_compute_derivatives,
    process_custom_derivatives_metadata,
    postprocess_compute_derivatives,
    preprocess_custom_inputs,
    postprocess_custom_outputs,
)

import warnings
import numpy as np

# warnings.simplefilter("always")

class CustomOperation(Operation):
    def __init__(self):
        self.input_dict = {}
        self.output_dict = {}
        self.derivative_parameters = {}
        self.name = 'custom'

class CustomExplicitOperation(CustomOperation):

    def __init__(self):
        """Wraps the evaluate method. No input arguments are required. 
        """

        super().__init__()

        self.evaluate = self._wrap_evaluate(self.evaluate)
        self.locked = False

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

            eval_outputs = evaluate(*args, **kwargs)
            Operation.__init__(self, *list(self.input_dict.values()))
            process_custom_derivatives_metadata(self.derivative_parameters, self.output_dict, self.input_dict)

            self.set_outputs(list(self.output_dict.values()))

            self.finalize_and_return_outputs()

            return eval_outputs
        return new_evaluate

    def compute_forward(self, inputs, outputs):
        self.compute(inputs, outputs)

    def compute_inline(self, *args):
        inputs = {key:input for key, input in zip(self.input_dict.keys(), args)}
        comp_outputs = {}

        inputs = preprocess_custom_inputs(inputs)
        self.compute_forward(inputs, comp_outputs)
        comp_outputs = postprocess_custom_outputs(comp_outputs, self.output_dict)

        output = [comp_outputs[key] for key in self.output_dict.keys()]
        if len(output) == 1:
            output = output[0]
        return output
    
    def compute_jax(self, *args):
        import jax

        def new_inline_func(*args):
            processed_inputs = [np.array(input) for input in args]
            return self.compute_inline(*processed_inputs)

        output = jax.pure_callback(
            new_inline_func,
            [jax.ShapeDtypeStruct(self.output_dict[output_var].shape, np.float64) for output_var in self.output_dict],
            *args)
        # if len(output) == 1:
        #     output = output[0]
        return tuple(output)

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
        rows=None,
        cols=None,
        val=None,
        method='exact', 
        step=None,
        form=None,
        step_calc=None,
        sparse = False,
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
        sparse : bool, optional
            Whether the user computed derivative is sparse (scipy sparse), by default False.

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
                        sparse=sparse,
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
                    sparse=sparse,
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
                    sparse=sparse,
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
            self.derivative_parameters[of, wrt]['sparse'] = sparse
    
    def evaluate_vjp(self, cotangents, *inputs_and_outputs):

        inputs = inputs_and_outputs[:self.num_inputs]
        outputs = inputs_and_outputs[self.num_inputs:]

        output_cots = []
        for output in outputs:
            if not cotangents.check(output):
                output_cots.append(Variable(value = np.zeros(output.shape)))
            else:
                output_cots.append(cotangents[output])
        input_cots = []
        for input in inputs:
            if cotangents.check(input):
                input_cots.append(input)

        vjps = self.build_custom_operation_vjp(
            input_cotangents = input_cots,
            output_cotangents = output_cots,
            deriv_order = 1)
        cots = vjps.finalize_and_return_outputs()
        
        if not isinstance(cots, tuple):
            cots = (cots,)
        for i, input in enumerate(input_cots):
            cotangents.accumulate(input, cots[i])

    def build_custom_operation_vjp(
            self,
            input_cotangents:list[Variable],
            output_cotangents:list[Variable],
            deriv_order:int)->'CustomJacOperation':

        if deriv_order > 1:
            raise NotImplementedError('Higher order custom derivatives not yet implemented')

        return CustomJacOperation(self, input_cotangents, output_cotangents, deriv_order)

class CustomJacOperation(Operation):
    
    def __init__(
            self,
            custom_operation:CustomOperation,
            cotangent_inputs:list[Variable],
            cotangent_outputs:list[Variable],
            order:int)->'CustomJacOperation':
        
        # forward inputs
        self.orig_inputs = custom_operation.inputs
        self.num_orig_inputs = len(self.orig_inputs)
        self.reverse_input_dict:dict[Variable,str] = {var:key for key,var in custom_operation.input_dict.items()}
        
        # forward outputs
        self.orig_outputs = custom_operation.outputs
        self.num_orig_outputs = len(self.orig_outputs)
        self.reverse_output_dict:dict[Variable,str] = {var:key for key,var in custom_operation.output_dict.items()}
        
        # cotangents
        self.input_cotangents = cotangent_inputs
        self.cotangents_outputs = cotangent_outputs
        
        # derivative order and original operation
        self.order = order
        self.custom_operation = custom_operation
        self.cache_jac:bool = True
        self.cached_inputs = None
        self.cached_jacs = None

        # The inputs of the operation are all the orginal inputs, the computed outputs AND the cotangents of the outputs
        vjp_custom_inputs = self.orig_inputs + self.orig_outputs + self.cotangents_outputs
        super().__init__(*vjp_custom_inputs)
        self.name = f'custom_jac_{order}'

        # We only want to output the cotangents that are actually necessary
        self.set_dense_outputs([input.shape for input in self.input_cotangents])

    def compute_inline(self, *orig_inputs_and_outputs_and_cots:list[np.array])->list[np.array]:
        """Perform the derivative accumulation procedure here.
        Two main steps:
        1. Call the user's compute_derivatives method and retrieve jacobians
        -- If the original input values are the same as the previous execution, we can use the previous jacobians
        2. Accumulate the cotangents using simple matrix vector products
        """
        input_values:list[np.array] = orig_inputs_and_outputs_and_cots[:self.num_orig_inputs]
        output_values:list[np.array] = orig_inputs_and_outputs_and_cots[self.num_orig_inputs:self.num_orig_inputs + self.num_orig_outputs]
        cot_values:list[np.array] = orig_inputs_and_outputs_and_cots[self.num_orig_inputs + self.num_orig_outputs:]
        if (not self.cache_jac) or (self.cached_inputs is None or not all(np.array_equal(input, cached_input) for input, cached_input in zip(input_values, self.cached_inputs))):
            inputs:dict[str,Variable] = {self.reverse_input_dict[key]:input for key, input in zip(self.orig_inputs, input_values)}
            outputs:dict[str,Variable] = {self.reverse_output_dict[key]:output for key, output in zip(self.orig_outputs, output_values)}

            # Call user derivatives
            derivatives_dict = prepare_compute_derivatives(self.custom_operation.derivative_parameters)
            inputs = preprocess_custom_inputs(inputs)
            outputs = preprocess_custom_inputs(outputs)
            self.custom_operation.compute_derivatives(inputs, outputs, derivatives_dict)
            postprocess_compute_derivatives(derivatives_dict, self.custom_operation.derivative_parameters)

            self.cached_inputs = input_values
            self.cached_jacs = derivatives_dict
        else:
            derivatives_dict = self.cached_jacs
        
        # Accumulate and return
        input_cots:list[np.array] = []
        for input in self.input_cotangents:
            # The cotangents are assumed to 2D vectors for now
            input_cots.append(np.zeros((1,input.size)))
            for i, output in enumerate(self.orig_outputs):
                output_str = self.reverse_output_dict[output]
                input_str = self.reverse_input_dict[input]

                cot_vector = cot_values[i].reshape(1, output.size)
                deriv_matrix = (derivatives_dict[output_str, input_str]).reshape(output.size, input.size)

                # for debugging:
                # print(output.name, input.name, input_cots[-1], cot_vector, deriv_matrix)
                input_cots[-1] += cot_vector@deriv_matrix

            # When we return it, it needs to be the correct shape of the input
            input_cots[-1] = input_cots[-1].reshape(input.shape)

        if len(input_cots) == 1:
            return input_cots[0]
        else:
            return tuple(input_cots)
    
    def compute_jax(self, *args):
        import jax

        def new_inline_func(*args):
            processed_inputs = [np.array(input) for input in args]
            return self.compute_inline(*processed_inputs)

        output = jax.pure_callback(
            new_inline_func,
            [jax.ShapeDtypeStruct(in_cot.shape, np.float64) for in_cot in self.input_cotangents],
            *args)
        # if len(output) == 1:
        #     output = output[0]
        
        return tuple(output)