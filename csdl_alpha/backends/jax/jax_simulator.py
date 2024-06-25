from csdl_alpha.backends.simulator import SimulatorBase, Recorder
from csdl_alpha.backends.jax.graph_to_jax import create_jax_interface
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import get_type_string, listify_variables
import numpy as np

from typing import Optional, Union, Callable

class VarManager():
    def __init__(
            self,
            variables:list[Variable],
            vars_type:str):
        self.list = variables 
        self.vars:set[Variable] = set(variables)
        self.vars_type = vars_type

    def verify_type(self, key:Variable):
        if not isinstance(key, Variable):
            raise TypeError(f"Key must be a Variable. {get_type_string(key)} given.")

    def verify_valid_var(self, var:Variable):
        if var not in self.vars:
            raise KeyError(f"Variable {var} ({var.name}) not recognized as an {self.vars_type}. Ensure that the variable is added as an {self.vars_type} to the simulator to get/set its values.")
        
    def check_valid_var(self, var:Variable)->bool:
        return var in self.vars

    def __getitem__(self, key:Variable)->np.ndarray:
        self.verify_type(key)
        self.verify_valid_var(key)
        return key.value

    def __setitem__(self, key:Variable, value:np.ndarray):
        self.verify_type(key)
        self.verify_valid_var(key)
        key.value = value

class JaxSimulator(SimulatorBase):

    def __init__(
            self, 
            recorder:Recorder,
            output_saved:bool = False,
            additional_inputs:list[Variable] = None,
            additional_outputs:Optional[list[Variable]] = None,
            gpu:bool = True,
            f64:bool = True,
            derivatives_kwargs:Optional[Union[dict[str],Callable]] = None,
        ):
        """Interface to run a recorder model using JAX. Variables of interest outside
        of optimization problems must be specified as additional inputs or outputs.

        Parameters
        ----------
        recorder : Recorder
            The recorder object with the model to run.
        output_saved : bool, optional
            If true, adds all variables that have been saved, by default False
        additional_inputs : list[Variable], optional
            A list of variables to specify as inputs in addition to design variables, by default None
        additional_outputs : Optional[list[Variable]], optional
            A list of variables to specify as outputs in addition to constraints and objectives, by default None
        gpu : bool, optional
            If True, compiles the jax function for GPU. If False or a GPU is not found, compiles for CPU. by default True
        f64 : bool, optional
            (Not yet implemented) If True, compiles the jax function for float64, otherwise compiles for float32. by default True
        """

        super().__init__(recorder)
        self.recorder:Recorder = recorder
        self.run_func:Optional[callable] = None
        self.totals_derivs:Optional[callable] = None
        self.run_forward_func:Optional[callable] = None
        self.opt_derivs_func:Optional[callable] = None

        # Process inputs and outputs
        if additional_inputs is None:
            additional_inputs = []
        self.additional_inputs:list[Variable] = listify_variables(additional_inputs)

        if additional_outputs is None:
            additional_outputs = []
        self.additional_outputs:list[Variable] = listify_variables(additional_outputs)

        # Gather saved outputs
        if not isinstance(output_saved, bool):
            raise TypeError(f"output_saved must be a bool. {get_type_string(output_saved)} given.")
        if output_saved:
            self.saved_outputs:list[Variable] = [node for node in self.recorder.active_graph.node_table if isinstance(node, Variable) and node._save]
        else:
            self.saved_outputs:list[Variable] = []

        # Process compilation settings
        if not isinstance(gpu, bool):
            raise TypeError(f"gpu must be a bool. {get_type_string(gpu)} given.")
        if not isinstance(f64, bool):
            raise TypeError(f"f64 must be a bool. {get_type_string(f64)} given.")
        self._gpu = 'gpu' if gpu else 'cpu'
        self._f64 = f64

        # Store valid inputs and outputs
        run_inputs = list(self.recorder.design_variables.keys())+self.additional_inputs
        run_outputs = list(self.recorder.objectives.keys())+list(self.recorder.constraints.keys())+self.saved_outputs+self.additional_outputs

        # Check to make sure inputs are valid:
        for input_var in run_inputs:
            if recorder.active_graph.in_degree(input_var) > 0:
                raise ValueError(f"Variable '{input_var}' (with name '{input_var.name}') is not a valid input. Only independent variables can be set as inputs")

        # Check to make sure outputs are valid:
        if len(run_outputs) == 0:
            raise ValueError("No outputs found. At least one objective, constraint, saved output, additional output must be specified.")
        if len(run_inputs) == 0:
            raise ValueError("No inputs found. At least one design variable or additional input must be specified.")

        self.input_manager:VarManager = VarManager(run_inputs, vars_type='input')
        self.output_manager:VarManager = VarManager(run_outputs, vars_type='output')

        # Process derivative kwargs
        if derivatives_kwargs is None:
            derivatives_kwargs = {}
        elif not isinstance(derivatives_kwargs, dict):
            raise TypeError(f"derivatives_kwargs must be a dict. {get_type_string(derivatives_kwargs)} given.")
        self.derivatives_kwargs:dict[str,any] = derivatives_kwargs
        self.derivatives_kwargs['as_block'] = False

    def run(self)->dict[Variable, np.ndarray]:
        """
        Computes all constraints, objectives, additional outputs and saved outputs
        """
        # Compile the function if it doesn't exist
        if self.run_func is None:
            print("compiling 'run' function ...")
            self.run_func = create_jax_interface(
                self.input_manager.list,
                self.output_manager.list,
                self.recorder.active_graph,
                device = self._gpu,
            )

        # Run the function
        outputs = self.run_func({in_var:in_var.value for in_var in self.input_manager.list})
        
        # Update the values 
        for output in outputs:
            output.set_value(outputs[output])

        return outputs

    def __getitem__(self, key:Variable)->np.ndarray:
        self.output_manager.verify_type(key)
        if self.output_manager.check_valid_var(key):
            return self.output_manager[key]
        elif self.input_manager.check_valid_var(key):
            return self.input_manager[key]
        else:
            # raise KeyError(f"Variable {key} ({key.name}) not an input or output. Add the variable as an additional input or output to the simulator in order to access values")
            raise KeyError(f"The variable '{key}' (with name '{key.name}') is not recognized as an input or output. Ensure that the variable is added as an input or output to the simulator to access its values.")
    def __setitem__(self, key:Variable, value:np.ndarray):
        self.input_manager[key] = value

        # regenerate run_forward, compute_optimization_derivatives if the updated variable is not an optimization input
        if key not in self.recorder.design_variables:
            self.run_forward_func = None
            self.opt_derivs_func = None

    def compute_totals(
            self,
            use_finite_difference:bool = False,
            finite_difference_step_size:float = 1e-6,
        )->dict[tuple[Variable, Variable], np.ndarray]:
        """
        Computes the total derivatives of all outputs with respect to all inputs
        """
        if not use_finite_difference:
            if self.totals_derivs is None:
                print("compiling 'compute_totals' function ...")

                import csdl_alpha as csdl
                self.recorder.start()
                self.derivative_variables = csdl.derivative(
                    ofs = self.output_manager.list,
                    wrts = self.input_manager.list,
                    **self.derivatives_kwargs
                )
                self.recorder.stop()

                self.totals_derivs = create_jax_interface(
                    self.input_manager.list,
                    list(self.derivative_variables.values()),
                    self.recorder.active_graph,
                    device = self._gpu,
                )

            derivs = self.totals_derivs({in_var:in_var.value for in_var in self.input_manager.list})

            return_derivs = {}
            for key in self.derivative_variables:
                return_derivs[key] = derivs[self.derivative_variables[key]]
            return return_derivs
        else:
            from csdl_alpha.src.operations.derivatives.derivative_utils import finite_difference
            if self.run_func is None:
                self.run()
            return finite_difference(
                ofs = self.output_manager.list,
                wrts = self.input_manager.list,
                step_size = finite_difference_step_size,
                forward_evaluation=self.run_func)
    
    def check_totals(
            self, 
            step_size:float = 1e-6,
            print_results:bool = True,
            raise_on_error:bool = False,
        )-> dict[tuple[Variable, Variable], dict[str]]:
        """
        Checks the total derivatives of all outputs with respect to all inputs
        """
        from csdl_alpha.src.operations.derivatives.derivative_utils import verify_derivative_values

        analytical_derivs = self.compute_totals()
        finite_difference_derivs = self.compute_totals(
            use_finite_difference=True,
            finite_difference_step_size=step_size,
        )

        verify_dict = {}
        for i, output in enumerate(self.output_manager.list):
            for j, input in enumerate(self.input_manager.list):
                
                tag = ''
                if output in self.recorder.objectives.keys():
                    tag += 'obj'
                elif output in self.recorder.constraints.keys():
                    tag += 'con'
                tag += f'{output.shape},'
                
                if input in self.recorder.design_variables.keys():
                    tag += f'dv'
                tag += f'{input.shape},'
                verify_dict[output, input] = {
                    'value': analytical_derivs[output, input],
                    'fd_value': finite_difference_derivs[output, input],
                    'of_name': output.name if not output.name is None else str(f'out_{i}'),
                    'wrt_name':  input.name if not input.name is None else str(f'in_{j}'),
                    'tag': tag,
                }
        return verify_derivative_values(
            verify_dict,
            raise_on_error=raise_on_error,
            print_results=print_results,
        )

    def get_inputs(self)->dict[Variable, np.ndarray]:
        return {in_var:in_var.value for in_var in self.input_manager.list}
    
    def get_outputs(self)->dict[Variable, np.ndarray]:
        return {out_var:out_var.value for out_var in self.output_manager.list}

    #     obj+constraints+saved_outputs+additional_outputs = f(dvs+additional_inputs)

    # def compute_totals(of, wrt):
        # of in  obj+constraints+saved_outputs+additional_outputs else error
        # wrt in dvs+additional_inputs else error
        
    # def check_totals():


    # def __getitem__(self, key):
        # key in  obj+constraints+saved_outputs+additional_outputs else error

    # def __setitem__(self, key):
        # key in dvs+additional_inputs else error
        # if key in additional_inputs:
            # set recompile run_forward True

    def run_forward(self):
        self.check_if_optimization()

        if self.run_forward_func is None:
            print("compiling 'run_forward' function ...")
            self.run_forward_func = create_jax_interface(
                list(self.recorder.design_variables.keys()),
                list(self.recorder.objectives.keys())+list(self.recorder.constraints.keys()),
                self.recorder.active_graph,
                device = self._gpu,
            )

        outputs = self.run_forward_func({dv:dv.value for dv in self.recorder.design_variables})
        for output in outputs:
            output.set_value(outputs[output])
        
        return self._process_optimization_values()

    def compute_optimization_derivatives(self):
        self.check_if_optimization()

        if self.opt_derivs_func is None:
            print("compiling 'compute_optimization_derivatives' function ...")

            self.recorder.start()
            self.build_objective_constraint_derivatives()
            self.recorder.stop()

            opt_derivs = []
            opt_derivs += [self.objective_gradient] if self.objective_gradient is not None else []
            opt_derivs += [self.constraint_jacobian] if self.constraint_jacobian is not None else []

            self.opt_derivs_func = create_jax_interface(
                list(self.recorder.design_variables.keys()),
                opt_derivs,
                self.recorder.active_graph,
                device = self._gpu,
            )

        outputs = self.opt_derivs_func({dv:dv.value for dv in self.recorder.design_variables})
        
        if self.objective_gradient is None:
            return None, outputs[self.constraint_jacobian]
        elif self.constraint_jacobian is None:
            return outputs[self.objective_gradient], None
        else:
            return outputs[self.objective_gradient], outputs[self.constraint_jacobian]
        
    def save_external(self, filename:str, groupname:str):
        from csdl_alpha.src.data import save_h5py_variables
        save_h5py_variables(filename, groupname, self.saved_outputs)