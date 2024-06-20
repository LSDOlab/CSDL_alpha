from csdl_alpha.src.recorder import Recorder
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import get_type_string

from typing import Union
import numpy as np

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
            dv_maps, dscaler, dlower, dupper, d0 = build_opt_metadata(recorder.design_variables, 'd')
            c_maps, cscaler, clower, cupper, _ = build_opt_metadata(recorder.constraints, 'c')
            o_maps, oscaler = build_opt_metadata(recorder.objectives, 'o')

            self.dv_meta = dv_maps
            self.c_meta = c_maps
            self.o_meta = o_maps

            self.opt_metadata['d'] = (dscaler, dlower, dupper, d0)
            self.opt_metadata['c'] = (cscaler, clower, cupper)
            self.opt_metadata['o'] = oscaler

    def get_optimization_metadata(self)->tuple[
            tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray],
            tuple[np.ndarray,np.ndarray,np.ndarray],
            np.ndarray,
        ]:
        """_summary_

        Returns
        -------
        tuple[ tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray], tuple[np.ndarray,np.ndarray,np.ndarray], np.ndarray, ]
            (dscaler, dlower, dupper, d0), (cscaler, clower, cupper), oscaler
        """
        self.check_if_optimization()
        return self.opt_metadata['d'], self.opt_metadata['c'], self.opt_metadata['o']

    def check_if_optimization(self):
        if not self.is_opt:
            raise ValueError("A valid optimization problem must be specified")

    def run(self):
        raise NotImplementedError('run method not implemented')
    
    def run_forward(self):
        raise NotImplementedError('run_forward method not implemented')

    def compute_optimization_derivatives(self):
        raise NotImplementedError('compute_optimization_derivatives method not implemented')

    def update_design_variables(self, dv_vector:np.ndarray)->None:
        self.check_if_optimization()

        for var in self.dv_meta:
            var.value = dv_vector[self.dv_meta[var]['l_ind']:self.dv_meta[var]['u_ind']].reshape(var.shape)

    def build_objective_constraint_derivatives(self):
        import csdl_alpha as csdl
        if len(self.recorder.constraints) > 0:
            self.constraint_jacobian = csdl.derivative(
                list(self.recorder.constraints.keys()),
                list(self.recorder.design_variables.keys()),
                as_block=True,
            )
        else:
            self.constraint_jacobian = None

        if len(self.recorder.objectives) > 0:
            self.objective_gradient = csdl.derivative(
                list(self.recorder.objectives.keys()),
                list(self.recorder.design_variables.keys()),
                as_block=True,
            )
        else:
            self.objective_gradient = None

class PySimulator(SimulatorBase):
    def __init__(
            self, 
            recorder:Recorder,
            ):
        super().__init__(recorder)
        self.recorder:Recorder = recorder
        self.initialize_totals = False

    def run(self):
        self.recorder.execute()

    def run_forward(self)->tuple[np.ndarray,np.ndarray]:
        self.check_if_optimization()
        self.recorder.execute()

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

    def compute_optimization_derivatives(self):
        
        self.check_if_optimization()

        if self.initialize_totals is False:
            self.recorder.start()
            self.build_objective_constraint_derivatives()
            self.recorder.stop()

            if not self.recorder.inline:
                self.recorder.execute()

            self.initialize_totals = True
        else:
            self.recorder.execute()
        
        if self.objective_gradient is None:
            return None, self.constraint_jacobian.value
        elif self.constraint_jacobian is None:
            return self.objective_gradient.value, None
        else:
            return self.objective_gradient.value, self.constraint_jacobian.value

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
    )->tuple[dict, np.array, np.array, np.array, np.array]:
    if meta_type not in ['d','c','o']:
        raise ValueError(f"meta_type must be one of ['d','c','o']. {meta_type} given.")

    metadata = {}

    concat_size = sum([var.size for var in recorder_data])

    if not meta_type == 'o':
        if concat_size == 0:
            return metadata, None, None, None, None
        lower_vector = -np.inf*np.ones(concat_size)
        upper_vector = np.inf*np.ones(concat_size)
        val_vector = np.zeros(concat_size)
    else:
        if concat_size == 0:
            return metadata, None
    scaler_vector = np.ones(concat_size)

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
                    raise ValueError(f"Design variable {var} must have an initial value specified.")

        scaler = recorder_data[var][0]
        if scaler is not None:
            scaler_vector[l_ind:u_ind] = scaler.flatten()

    if not meta_type == 'o':
        return metadata, scaler_vector, lower_vector, upper_vector, val_vector
    else:
        return metadata, scaler_vector