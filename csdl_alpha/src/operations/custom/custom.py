from csdl_alpha.utils.parameters import Parameters
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.src.graph.operation import Operation
from csdl_alpha.src.graph.node import Node

class CustomModel(Operation):
    def __init__(self, *args, **kwargs):
        self.parameters = Parameters()
        self.parameters.hold(kwargs)
        self.initialize()
        self.parameters.check(kwargs)

        self.inputs = {}
        self.outputs = {}
        self.derivative_parameters = {}

class CustomExplicitModel(CustomModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.evaluate = self._wrap_evaluate(self.evaluate)

    def initialize(self):
        raise NotImplementedError('not implemented')

    def evaluate(self):
        raise NotImplementedError('not implemented')

    def compute(self, inputs, outputs):
        raise NotImplementedError('not implemented')

    def compute_derivatives(self, inputs, outputs, derivatives):
        raise NotImplementedError('not implemented')

    def _wrap_evaluate(self, evaluate):
        def new_evaluate(*args, **kwargs):
            Node.__init__(self)
            self.name = self.__class__.__name__

            self.recorder._add_node(self)
            eval_outputs = evaluate(*args, **kwargs)

            for input_var in self.inputs.values():
                self.recorder._add_edge(input_var, self)

            for output_var in self.outputs.values():
                self.recorder._add_edge(self, output_var)

            if self.recorder.inline:
                self.set_inline_values()

            return eval_outputs
        return new_evaluate

    def set_inline_values(self):
        inputs = {key: input.value for key, input in self.inputs.items()}
        comp_outputs = {}

        self.compute(inputs, comp_outputs)

        for key, output in self.outputs.items():
            output.value = comp_outputs[key]

    def create_output(self, name, shape):
            """Create and store an output variable.

            This method creates a new output variable with the given name and shape,
            and stores it in the `outputs` dictionary of the object.

            Parameters
            ----------
            name : str
                The name of the output variable.
            shape : tuple
                The shape of the output variable.

            Returns
            -------
            Variable
                The created output variable.
            """
            output = Variable(shape)
            self.outputs[name] = output
            return output
    
    def declare_input(self, key, variable):
        """Declares a variable as an input.

        This method stores the given input variable in the inputs dictionary under the given key.

        Parameters
        ----------
        key : str
            The key for the variable.
        variable : csdl.Variable
            The input variable.
        """
        self.inputs[key] = variable
    
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
                of_list = list(self.outputs.keys())
        elif isinstance(of, list):
            if any(x == '*' for x in of):
                of_list = list(self.outputs.keys())
            else:
                of_list = of
        if isinstance(wrt, str):
            if wrt == '*':
                wrt_list = list(self.inputs.keys())
        elif isinstance(wrt, list):
            if any(x == '*' for x in wrt):
                wrt_list = list(self.inputs.keys())
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

