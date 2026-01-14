from csdl_alpha.backends.simulator import SimulatorBase, Recorder, timer
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
            raise KeyError(f"Variable {var.info()} not recognized as an {self.vars_type}. Ensure that the variable is added as an {self.vars_type} to the simulator to get/set its values.")
        
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
            save_on_update:bool = False,
            filename:str = 'jax_simulator.hdf5',
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
        save_on_update : bool, optional - EXPERIMENTAL
            If True, saves the values of all saved outputs to an external file after each derivative call, by default False
        filename : str, optional - EXPERIMENTAL
            The name of the file to save the values of all saved outputs to, by default 'jax_simulator.hdf5'
        """

        super().__init__(recorder)
        self.recorder:Recorder = recorder
        self.run_func:Optional[callable] = None
        self.totals_derivs:Optional[callable] = None
        self.run_forward_func:Optional[callable] = None
        self.opt_derivs_func:Optional[callable] = None

        self.update_counter = 0
        self.save_on_update = save_on_update
        self.filename = filename
        self.callback_func = None

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
            self.saved_outputs:list[Variable] = [node for node in self.recorder.get_root_graph().node_table if isinstance(node, Variable) and node._save]
        else:
            self.saved_outputs:list[Variable] = []

        # Process compilation settings
        if not isinstance(gpu, bool):
            raise TypeError(f"gpu must be a bool. {get_type_string(gpu)} given.")
        if not isinstance(f64, bool):
            raise TypeError(f"f64 must be a bool. {get_type_string(f64)} given.")
        self._gpu = 'gpu' if gpu else 'cpu'
        self.use_f64 = f64

        # Store valid inputs and outputs
        run_inputs = list(self.recorder.design_variables.keys())+self.additional_inputs
        run_outputs = list(self.recorder.objectives.keys())+list(self.recorder.constraints.keys())+self.saved_outputs+self.additional_outputs

        # Even if output_saved is False, still want to save any variables in inputs/outputs
        if len(self.saved_outputs) == 0:
            self.saved_outputs = [var for var in run_inputs if var._save] + [var for var in run_outputs if var._save]

        # Check to make sure inputs are valid:
        for input_var in run_inputs:
            if recorder.get_root_graph().in_degree(input_var) > 0:
                raise ValueError(f"Variable {input_var.info()} is not a valid input. Only independent variables can be set as inputs")

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

    # @timer # For debugging
    def run(self)->dict[Variable, np.ndarray]:
        """
        Computes all constraints, objectives, additional outputs and saved outputs
        """
        # Compile the function if it doesn't exist
        if self.run_func is None:
            print(f"compiling 'run' function ... ({len(self.recorder.node_graph_map)} nodes)")
            self.run_func = create_jax_interface(
                self.input_manager.list,
                self.output_manager.list,
                self.recorder.get_root_graph(),
                device = self._gpu,
                enable_f64=self.use_f64,
                name = 'run',
            )

        # Run the function
        outputs = self.run_func({in_var:in_var.value for in_var in self.input_manager.list})
        
        # Callback if user-specified
        self.callback()

        # Update the values 
        for output in outputs:
            output.set_value(outputs[output])

        return outputs

    def __getitem__(self, key:Union[Variable, str])->np.ndarray:
        if isinstance(key, str):
            key = self.recorder._find_variables_by_name(key)[0]

        self.output_manager.verify_type(key)
        if self.output_manager.check_valid_var(key):
            return self.output_manager[key]
        elif self.input_manager.check_valid_var(key):
            return self.input_manager[key]
        else:
            # raise KeyError(f"Variable {key} ({key.name}) not an input or output. Add the variable as an additional input or output to the simulator in order to access values")
            raise KeyError(f"The variable {key.info()} is not recognized as an input or output. Ensure that the variable is added as an input or output to the simulator to access its values.")
    def __setitem__(self, key:Variable, value:np.ndarray):
        self.input_manager[key] = value

        # regenerate run_forward, compute_optimization_derivatives if the updated variable is not an optimization input
        if key not in self.recorder.design_variables:
            self.run_forward_func = None
            self.opt_derivs_func = None

    # @timer # For debugging
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

                import csdl_alpha as csdl
                self.recorder.start()
                self.derivative_variables = csdl.derivative(
                    ofs = self.output_manager.list,
                    wrts = self.input_manager.list,
                    **self.derivatives_kwargs
                )
                self.recorder.stop()
                print(f"compiling 'compute_totals' function ... ({len(self.recorder.node_graph_map)} nodes)")

                self.totals_derivs = create_jax_interface(
                    self.input_manager.list,
                    list(self.derivative_variables.values()),
                    self.recorder.get_root_graph(),
                    device = self._gpu,
                    enable_f64=self.use_f64,
                    name = 'compute_totals',
                )

            derivs = self.totals_derivs({in_var:in_var.value for in_var in self.input_manager.list})
            
            # Callback if user-specified
            self.callback()

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
        Checks the total derivatives of all outputs with respect to all inputs using finite difference
        """
        from csdl_alpha.src.operations.derivatives.derivative_utils import verify_derivative_values

        analytical_derivs = self.compute_totals()
        finite_difference_derivs = self.compute_totals(
            use_finite_difference=True,
            finite_difference_step_size=step_size,
        )

        verify_dict = self._build_check_totals_verification_dict(
            self.output_manager.list,
            self.input_manager.list,
            analytical_derivs,
            finite_difference_derivs,
        )

        return verify_derivative_values(
            verify_dict,
            raise_on_error=raise_on_error,
            print_results=print_results,
        )

    def get_inputs(self)->dict[Variable, np.ndarray]:
        return {in_var:in_var.value for in_var in self.input_manager.list}
    
    def get_outputs(self)->dict[Variable, np.ndarray]:
        return {out_var:out_var.value for out_var in self.output_manager.list}

    # @timer # For debugging
    def run_forward(self):
        self.check_if_optimization()

        if self.run_forward_func is None:
            print(f"compiling 'run_forward' function ... ({len(self.recorder.node_graph_map)} nodes)")
            self.run_forward_func = create_jax_interface(
                list(self.recorder.design_variables.keys()),
                list(self.recorder.objectives.keys())+list(self.recorder.constraints.keys())+self.saved_outputs,
                self.recorder.get_root_graph(),
                device = self._gpu,
                enable_f64=self.use_f64,
                name = 'run_forward',
            )

        outputs = self.run_forward_func({dv:dv.value for dv in self.recorder.design_variables})
        for output in outputs:
            output.set_value(outputs[output])
        
        # Callback if user-specified
        self.callback()

        return self._process_optimization_values()
    
    # @timer # For debugging
    def compute_optimization_derivatives(
            self,
            use_finite_difference:bool = False,
            finite_difference_step_size:float = 1e-6,
        )->dict[str,Variable]:
        if self.save_on_update:
            self.save_external(self.filename, 'iteration_'+str(self.update_counter))
            self.update_counter += 1

        self.check_if_optimization()
        
        # Initialize return dict
        return_dict = {
            "f": None,
            "c": None,
            "df": None,
            "dc": None,
        }

        if not use_finite_difference:
            if self.opt_derivs_func is None:

                self.recorder.start()
                self.build_objective_constraint_derivatives(self.derivatives_kwargs)
                self.recorder.stop()
                print(f"compiling 'compute_optimization_derivatives' function ... ({len(self.recorder.node_graph_map)} nodes)")

                opt_derivs = []
                opt_derivs += list(self.recorder.objectives.keys())
                opt_derivs += list(self.recorder.constraints.keys())
                opt_derivs += [self.objective_gradient] if self.objective_gradient is not None else []
                opt_derivs += [self.constraint_jacobian] if self.constraint_jacobian is not None else []

                self.opt_derivs_func = create_jax_interface(
                    list(self.recorder.design_variables.keys()),
                    opt_derivs,
                    self.recorder.get_root_graph(),
                    device = self._gpu,
                    enable_f64=self.use_f64,
                    name = 'compute_optimization_derivatives',
                )
            outputs = self.opt_derivs_func({dv:dv.value for dv in self.recorder.design_variables})
            for output in outputs:
                output.set_value(outputs[output])

            # fill out return_dict
            if self.objective_gradient is not None:
                return_dict["df"] = outputs[self.objective_gradient]
            if self.constraint_jacobian is not None:
                return_dict["dc"] = outputs[self.constraint_jacobian]
            
            # Callback if user-specified
            self.callback()

            # get optimization values
            f,c = self._process_optimization_values()
            return_dict["f"] = f
            return_dict["c"] = c
        else:
            from csdl_alpha.src.operations.derivatives.derivative_utils import finite_difference
            # fwd evaluation
            f,c = self.run_forward()
            
            # derivatives
            outputs = finite_difference(
                ofs = list(self.recorder.constraints.keys()) + list(self.recorder.objectives.keys()),
                wrts = list(self.recorder.design_variables.keys()),
                step_size = finite_difference_step_size,
                forward_evaluation=self.run_forward_func)
            df, dc = self._assemble_jacs(outputs)

            # fill out return_dict
            return_dict["f"] = f
            return_dict["c"] = c
            return_dict["df"] = df
            return_dict["dc"] = dc

        return return_dict
        
    def save_external(self, filename:str, groupname:str):
        from csdl_alpha.src.data import save_h5py_variables
        # update variable values
        for output in self.saved_outputs:
            output.value = self[output]
        save_h5py_variables(filename, groupname, self.saved_outputs)

    def add_callback(self, func:Callable):
        self.callback_func = func

    def callback(self):
        if self.callback_func is not None:
            self.callback_func(self)
