from csdl_alpha.src.recorder import Recorder
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import get_type_string, listify_variables

from typing import Optional, Union, Callable
import numpy as np

# For debugging:
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        print(f"START {func.__name__}")
        s = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"END {func.__name__} took {end-s} seconds")
        return result
    return wrapper

class SimulatorBase():
    def __init__(
            self, 
            recorder:Recorder,
            ):
        if not isinstance(recorder, Recorder):
            raise TypeError(f"recorder must be an instance of Recorder. {get_type_string(recorder)} given.")
        
        self.initialized_totals = False
        
        self.recorder:Recorder = recorder

        self.is_opt = determine_if_optimization(recorder)

        if self.is_opt:
            self.opt_metadata:dict[str,dict[Variable,Union[np.array]]] = {}
            dv_maps, dscaler, dlower, dupper, d0, dadder = build_opt_metadata(recorder.design_variables, 'd')
            c_maps, cscaler, clower, cupper, _, cadder = build_opt_metadata(recorder.constraints, 'c')
            o_maps, oscaler, oadder = build_opt_metadata(recorder.objectives, 'o')

            self.dv_meta = dv_maps
            self.c_meta = c_maps
            self.o_meta = o_maps

            self.opt_metadata['d'] = (dscaler, dlower, dupper, d0, dadder)
            self.opt_metadata['c'] = (cscaler, clower, cupper, cadder)
            self.opt_metadata['o'] = (oscaler, oadder)

    def get_optimization_metadata(self)->tuple[
            tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray],
            tuple[np.ndarray,np.ndarray,np.ndarray],
            np.ndarray,
        ]:
        """_summary_

        Returns
        -------
        tuple[ tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray], tuple[np.ndarray,np.ndarray,np.ndarray], np.ndarray, ]
            (dscaler, dlower, dupper, d0, dadder), (cscaler, clower, cupper, cadder), (oscaler, oadder)
        """
        self.check_if_optimization()
        return self.opt_metadata['d'], self.opt_metadata['c'], self.opt_metadata['o']

    def check_if_optimization(self):
        if not self.is_opt:
            raise ValueError(f"A valid optimization problem must be specified to use this method. {len(self.recorder.design_variables)} design variables, {len(self.recorder.objectives)} objectives, and {len(self.recorder.constraints)} constraints found.")

    def run(self):
        raise NotImplementedError('run method not implemented')
    
    def run_forward(self)->tuple[np.ndarray]:
        raise NotImplementedError('run_forward method not implemented')

    def compute_optimization_derivatives(self)->dict[str,np.ndarray]:
        raise NotImplementedError('compute_optimization_derivatives method not implemented')
    
    def compute_optimization_functions(self)->dict[str,np.ndarray]:
        f,c = self.run_forward()
        return {"f": f, "c": c}

    def update_design_variables(self, dv_vector:np.ndarray, save = False)->None:
        self.check_if_optimization()

        for var in self.dv_meta:
            var.value = dv_vector[self.dv_meta[var]['l_ind']:self.dv_meta[var]['u_ind']].reshape(var.shape)

        if save:
            self.run()
            self.recorder.inline_save()
    def build_objective_constraint_derivatives(self, derivative_kwargs):

        if not self.initialized_totals:
            self.initialized_totals = True
            import csdl_alpha as csdl

            derivative_kwargs = derivative_kwargs.copy()
            derivative_kwargs['as_block'] = True
            opt_jac = csdl.derivative(
                    list(self.recorder.objectives.keys())+list(self.recorder.constraints.keys()),
                    list(self.recorder.design_variables.keys()),
                    **derivative_kwargs,
                )
            
            num_dvs = sum([var.size for var in self.recorder.design_variables])
            if len(self.recorder.objectives) > 0:
                self.objective_gradient = opt_jac[0,:].reshape((1, num_dvs))
                if len(self.recorder.constraints) > 0:
                    num_cvs = sum([var.size for var in self.recorder.constraints])
                    self.constraint_jacobian = opt_jac[1:,:].reshape((num_cvs, num_dvs))
                else:
                    self.constraint_jacobian = None
            else:
                self.objective_gradient = None
                if len(self.recorder.constraints) > 0:
                    self.constraint_jacobian = opt_jac
                else:
                    self.constraint_jacobian = None

    def _process_optimization_values(self):
        nc = sum([var.size for var in self.recorder.constraints])
        if nc > 0:
            constraints = np.zeros((sum([var.size for var in self.recorder.constraints]),))
            for var in self.c_meta:
                constraints[self.c_meta[var]['l_ind']:self.c_meta[var]['u_ind']] = var.value.flatten()
        else:
            constraints = None
        
        no = sum([var.size for var in self.recorder.objectives])
        if no > 0:
            objectives = np.zeros((sum([var.size for var in self.recorder.objectives]),))
            for var in self.o_meta:
                objectives[self.o_meta[var]['l_ind']:self.o_meta[var]['u_ind']] = var.value.flatten()
        else:
            objectives = None

        return objectives, constraints
    
    def _build_check_totals_verification_dict(
            self,
            ofs:list[Variable],
            wrts:list[Variable],
            analytical:dict[tuple[Variable, Variable], np.ndarray],
            fd:dict[tuple[Variable, Variable], np.ndarray]
        )->dict[tuple[Variable, Variable], dict]:
        
        verify_dict = {}
        for i, output in enumerate(ofs):
            for j, input in enumerate(wrts):
                
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
                    'value': analytical[output, input],
                    'fd_value': fd[output, input],
                    'of_name': output.name if not output.name is None else str(f'out_{i}'),
                    'wrt_name':  input.name if not input.name is None else str(f'in_{j}'),
                    'tag': tag,
                }
        return verify_dict

    def _assemble_jacs(self, derivs:dict[tuple[Variable, Variable], np.ndarray])->tuple[np.ndarray]:
        """returns a tuple of the objective gradient and constraint gradient

        Parameters
        ----------
        derivs : dict[tuple[Variable, Variable], np.ndarray]

        Returns
        -------
        tuple[np.ndarray]: df, dc
        """
        num_dvs = self.opt_metadata['d'][0].size

        if len(self.recorder.objectives) == 0:
            objective_grad = None
        else:
            objective = list(self.recorder.objectives.keys())[0]
            objective_grad = np.zeros((objective.size, num_dvs))
            for dv in self.dv_meta:
                lower_index = self.dv_meta[dv]['l_ind']
                upper_index = self.dv_meta[dv]['u_ind']
                objective_grad[0, lower_index:upper_index] = derivs[objective, dv].flatten()

        if len(self.recorder.constraints) == 0:
            constraint_jac = None
        else:
            num_cvs = self.opt_metadata['c'][0].size
            constraint_jac = np.zeros((num_cvs, num_dvs))
            for constraint in self.recorder.constraints:
                cl_index = self.c_meta[constraint]['l_ind']
                cu_index = self.c_meta[constraint]['u_ind']
                for dv in self.recorder.design_variables:
                    dl_index = self.dv_meta[dv]['l_ind']
                    du_index = self.dv_meta[dv]['u_ind']
                    constraint_jac[cl_index:cu_index, dl_index:du_index] = derivs[constraint, dv]

        return objective_grad, constraint_jac

    def _unassemble_jacs(self, objective_grad:np.ndarray, constraint_jac:np.ndarray)->dict[tuple[Variable, Variable], np.ndarray]:
        """returns a dictionary of the derivatives given constraint and objective gradients

        Parameters
        ----------
        objective_grad : np.ndarray
        constraint_jac : np.ndarray

        Returns
        -------
        dict[tuple[Variable, Variable], np.ndarray]
        """
        derivs = {}
        if objective_grad is not None:
            objective = list(self.recorder.objectives.keys())[0]
            for dv in self.dv_meta:
                lower_index = self.dv_meta[dv]['l_ind']
                upper_index = self.dv_meta[dv]['u_ind']
                derivs[objective, dv] = objective_grad[0, lower_index:upper_index].reshape(1, dv.size)

        if constraint_jac is not None:
            for constraint in self.recorder.constraints:
                cl_index = self.c_meta[constraint]['l_ind']
                cu_index = self.c_meta[constraint]['u_ind']
                for dv in self.recorder.design_variables:
                    dl_index = self.dv_meta[dv]['l_ind']
                    du_index = self.dv_meta[dv]['u_ind']
                    derivs[constraint, dv] = constraint_jac[cl_index:cu_index, dl_index:du_index].reshape(constraint.size, dv.size)

        return derivs

    def check_optimization_derivatives(
            self, 
            step_size:float = 1e-6,
            print_results:bool = True,
            raise_on_error:bool = False,
        ):
        """
        Checks the total derivatives of all optimization outputs with respect to all design variables using finite difference
        """
        from csdl_alpha.src.operations.derivatives.derivative_utils import verify_derivative_values

        # Analytical:
        ads = self.compute_optimization_derivatives()
        analytical_derivs = self._unassemble_jacs(ads['df'], ads['dc'])
        
        # Finite difference:
        fds = self.compute_optimization_derivatives(
            use_finite_difference=True,
            finite_difference_step_size=step_size,
        )
        finite_difference_derivs = self._unassemble_jacs(fds['df'], fds['dc'])

        verify_dict = self._build_check_totals_verification_dict(
            list(self.recorder.objectives.keys())+list(self.recorder.constraints.keys()),
            list(self.recorder.design_variables.keys()),
            analytical_derivs,
            finite_difference_derivs,
        )

        return verify_derivative_values(
            verify_dict,
            raise_on_error=raise_on_error,
            print_results=print_results,
        )


class PySimulator(SimulatorBase):
    def __init__(
            self, 
            recorder:Recorder,
            derivatives_kwargs:Optional[Union[dict[str],Callable]] = None,
            ):
        super().__init__(recorder)
        self.recorder:Recorder = recorder
        self.initialize_totals = False

        self.derivative_hashes = {}
        self.derivative_kwargs = derivatives_kwargs if derivatives_kwargs is not None else {}

    # @timer
    def run(self):
        self.recorder.execute()

    # @timer
    def run_forward(self)->tuple[np.ndarray,np.ndarray]:
        self.check_if_optimization()
        self.recorder.execute()
        return self._process_optimization_values()

    # @timer
    def compute_optimization_derivatives(
            self,
            use_finite_difference:bool = False,
            finite_difference_step_size:float = 1e-6,
        )->dict[str,Variable]:
        
        self.check_if_optimization()

        # Initialize return dict
        return_dict = {
            "f": None,
            "c": None,
            "df": None,
            "dc": None,
        }

        if not use_finite_difference:
            if self.initialize_totals is False:
                self.recorder.start()
                self.build_objective_constraint_derivatives(self.derivative_kwargs)
                self.recorder.stop()

                if not self.recorder.inline:
                    self.recorder.execute()

                self.initialize_totals = True
            else:
                self.recorder.execute()
            
            # fill out return_dict
            if self.objective_gradient is not None:
                return_dict["df"] = self.objective_gradient.value
            if self.constraint_jacobian is not None:
                return_dict["dc"] = self.constraint_jacobian.value
            
            # get optimization values
            f,c = self._process_optimization_values()
            return_dict["f"] = f
            return_dict["c"] = c

        else:
            from csdl_alpha.src.operations.derivatives.derivative_utils import finite_difference
            
            def forward_evaluation(wrts_arg:dict[Variable, np.array])->dict[Variable, np.array]:
                graph = self.recorder.active_graph
                for wrt in wrts_arg:
                    wrt.value = wrts_arg[wrt]
                graph.execute_inline()
                return {of:of.value for of in list(self.recorder.constraints.keys()) + list(self.recorder.objectives.keys())}

            outputs = finite_difference(
                ofs = list(self.recorder.constraints.keys()) + list(self.recorder.objectives.keys()),
                wrts = list(self.recorder.design_variables.keys()),
                step_size = finite_difference_step_size,
                forward_evaluation=forward_evaluation,
            )
            f, c = self._process_optimization_values()
            df, dc = self._assemble_jacs(outputs)

            # fill out return_dict
            return_dict["f"] = f
            return_dict["c"] = c
            return_dict["df"] = df
            return_dict["dc"] = dc

        return return_dict

    def __getitem__(self, key:Variable):
        if not isinstance(key, Variable):
            raise KeyError(f"{key} must be an instance of Variable. {get_type_string(key)} given.")
        return key.value

    def __setitem__(self, key:Variable, value:np.ndarray):
        if not isinstance(key, Variable):
            raise KeyError(f"{key} must be an instance of Variable. {get_type_string(key)} given.")
        if self.recorder.get_root_graph().in_degree(key) > 0:
            raise ValueError(f"Variable {key.info()} is not a valid input. Only independent variables can be set as inputs")

        key.value = value

    # @timer
    def compute_totals(
            self,
            ofs:list[Variable],
            wrts:list[Variable],
            use_finite_difference:bool = False,
            finite_difference_step_size:float = 1e-6,
        )->dict[tuple[Variable, Variable], np.ndarray]:
        """
        Computes the total derivatives of all outputs with respect to all inputs
        """
        ofs = listify_variables(ofs)
        wrts = listify_variables(wrts)
        if not use_finite_difference:
            hash_key = (tuple(ofs), tuple(wrts))
            
            if not hash_key in self.derivative_hashes:
                import csdl_alpha as csdl
                self.recorder.start()
                self.derivative_hashes[hash_key] = csdl.derivative(
                    ofs = ofs,
                    wrts = wrts,
                    **self.derivative_kwargs,
                )
                self.recorder.stop()

            self.recorder.execute()

            return_derivs = {}
            for of in ofs:
                for wrt in wrts:
                    return_derivs[of, wrt] = self.derivative_hashes[hash_key][of, wrt].value
            return return_derivs
        else:
            from csdl_alpha.src.operations.derivatives.derivative_utils import finite_difference
            
            def forward_evaluation(wrts_arg:dict[Variable, np.array])->dict[Variable, np.array]:
                graph = self.recorder.active_graph
                for wrt in wrts_arg:
                    wrt.value = wrts_arg[wrt]
                graph.execute_inline()
                return {of:of.value for of in ofs}
            
            return finite_difference(
                ofs = ofs,
                wrts = wrts,
                step_size = finite_difference_step_size,
                forward_evaluation=forward_evaluation)
        
    def check_totals(
            self,
            ofs:list[Variable] = None,
            wrts:list[Variable] = None,
            step_size:float = 1e-6,
            print_results:bool = True,
            raise_on_error:bool = False,
        ):
        """
        Checks the total derivatives of ofs with respect to wrts using finite difference
        """
        if ofs is None:
            ofs = list(self.recorder.objectives.keys()) + list(self.recorder.constraints.keys())
            if len(ofs) == 0:
                raise ValueError("No objectives or constraints found. Add objectives/constraints or specify 'ofs'.")
        if wrts is None:
            wrts = list(self.recorder.design_variables.keys())
            if len(wrts) == 0:
                raise ValueError("No design variables found. Add design variables or specify 'wrts'.")

        from csdl_alpha.src.operations.derivatives.derivative_utils import verify_derivative_values
        ofs = listify_variables(ofs)
        wrts = listify_variables(wrts)
        analytical_derivs = self.compute_totals(
            ofs,
            wrts,
        )
        finite_difference_derivs = self.compute_totals(
            ofs,
            wrts,
            use_finite_difference=True,
            finite_difference_step_size=step_size,
        )

        verify_dict = self._build_check_totals_verification_dict(
            ofs,
            wrts,
            analytical_derivs,
            finite_difference_derivs,
        )

        return verify_derivative_values(
            verify_dict,
            raise_on_error=raise_on_error,
            print_results=print_results,
        )

def determine_if_optimization(recorder:Recorder)->bool:
    """
    Determine if the recorder specifies an optimization problem
    """
    if len(recorder.design_variables) > 0 and (len(recorder.objectives)+len(recorder.constraints)) > 0:
        return True
    else:
        return False


def build_opt_metadata(
        recorder_data:dict[Variable,Union[np.array]],
        meta_type:str   
    )->tuple[dict, np.array, np.array, np.array, np.array, np.array]:
    if meta_type not in ['d','c','o']:
        raise ValueError(f"meta_type must be one of ['d','c','o']. {meta_type} given.")

    metadata = {}

    concat_size = sum([var.size for var in recorder_data])

    if not meta_type == 'o':
        if concat_size == 0:
            return metadata, None, None, None, None, None
        lower_vector = -np.inf*np.ones(concat_size)
        upper_vector = np.inf*np.ones(concat_size)
        val_vector = np.zeros(concat_size)
    else:
        if concat_size == 0:
            return metadata, None, None
    scaler_vector = np.ones(concat_size)
    adder_vector  = np.zeros(concat_size)

    running_size = 0
    for var in recorder_data:
        l_ind = running_size
        running_size += var.size
        u_ind = running_size

        metadata[var] = {}
        metadata[var]['l_ind'] = l_ind
        metadata[var]['u_ind'] = u_ind

        if not meta_type == 'o':
            lower = recorder_data[var][1]
            if lower is not None:
                lower_vector[l_ind:u_ind] = lower.flatten()

            upper = recorder_data[var][2]
            if upper is not None:
                upper_vector[l_ind:u_ind] = upper.flatten()
            
            val = var.value
            if val is not None:
                val_vector[l_ind:u_ind] = val.flatten()
            else:
                if meta_type == 'd':
                    raise ValueError(f"Design variable {var.info()} must have an initial value specified.")

        scaler = recorder_data[var][0]
        if scaler is not None:
            scaler_vector[l_ind:u_ind] = scaler.flatten()

        adder = recorder_data[var][-1]
        if adder is not None:
            adder_vector[l_ind:u_ind] = adder.flatten()

    if not meta_type == 'o':
        return metadata, scaler_vector, lower_vector, upper_vector, val_vector, adder_vector
    else:
        return metadata, scaler_vector, adder_vector
    