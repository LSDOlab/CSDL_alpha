from csdl_alpha.src.operations.implicit_operations.nonlinear_solvers.nonlinear_solver import NonlinearSolver, check_variable_shape_compatibility, check_run_time_tolerance
from csdl_alpha.src.operations.implicit_operations.nonlinear_solvers.gauss_seidel import GaussSeidel
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import scalarize, ingest_value
import numpy as np

from typing import Union


class Jacobi(GaussSeidel):
    
    def __init__(
            self,
            name = 'jacobi_nlsolver',
            print_status = True,
            tolerance=1e-10,
            max_iter=100,
        ):
        """
        A nonlinear block Jacobi solver

        The states are updated all at once:
        x0_new = x0_state_update(x0_old, x1_old, ... xn_old)
        x1_new = x1_state_update(x0_old, x1_old, ... xn_old)
        ...
        xn_new = xn_state_update(x0_old, x1_old, ... xn_old)
        """
        super().__init__(
            name = name,
            print_status = print_status,
            tolerance = tolerance,
            max_iter = max_iter
        )

    def _inline_update_states(self):
        # while not converged:
        #    x0_new = x0_state_update(x0_old, x1_old, ... xn_old)
        #    x1_new = x1_state_update(x0_old, x1_old, ... xn_old)
        #    ...
        #    xn_new = xn_state_update(x0_old, x1_old, ... xn_old)
        
        # compute residuals once before hand
        for current_state, current_residual in self.state_to_residual_map.items():
            # get current state value and residual value
            current_state_value = current_state.value
            current_residual_value = current_residual.value

            # update current state value
            if self.state_metadata[current_state]['state_update'] is None:
                current_state.value = current_state_value - current_residual_value
            else:
                current_state.value = self.state_metadata[current_state]['state_update'].value