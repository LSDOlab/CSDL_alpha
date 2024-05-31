from csdl_alpha.src.operations.implicit_operations.nonlinear_solvers.nonlinear_solver import NonlinearSolver, check_variable_shape_compatibility, check_run_time_tolerance
from csdl_alpha.src.operations.implicit_operations.nonlinear_solvers.fixed_point import FixedPoint
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import scalarize, ingest_value
import numpy as np

from typing import Union

class GaussSeidel(FixedPoint):
    
    def __init__(
            self,
            name = 'gs_nlsolver',
            print_status = True,
            tolerance=1e-10,
            max_iter=100,
            elementwise_states=False,
        ):
        """
        A nonlinear block Gauss-Seidel solver

        The states are updated after each other in the order they are added:
        x0_new = x0_state_update(x0_old, x1_old, ... xn_old)
        x1_new = x1_state_update(x0_new, x1_old, ... xn_old)
        ...
        xn_new = xn_state_update(x0_new, x1_new, ... xn_old)
        """
        # well now this just seems redundant
        super().__init__(
            name = name,
            print_status = print_status,
            tolerance = tolerance,
            max_iter = max_iter,
            elementwise_states = elementwise_states,
        )

    def add_state(
            self,
            state: Variable,
            residual: Variable,
            state_update:Variable = None,
            initial_value: Union[Variable, float] = None,
            tolerance: Union[Variable, float] = None
        ):
        """
        state: Variable
            The state variable to be solved for.
        residual: Variable
            The residual variable that should be minimized.
        state_update: Variable
            The update to the state variable. If None, it is automatically calculated.
        initial_value: Variable or float
            The initial value of the state variable. If None, it is set to the current value.
        tolerance: Variable or float
            The tolerance for the state variable. If None, it is set to the solver's tolerance.
        """

        # add the state and residual pair
        self.add_state_residual_pair(state, residual)

        # Store metadata about the state
        if state_update is not None:
            if not isinstance(state_update, Variable):
                raise ValueError("State update must be a Variable")
            self.add_intersection_target(state_update)

        self.add_state_metadata(state, 'state_update', state_update, is_input=False)

        # Check if user provided an initial value
        if initial_value is None:
            self.add_state_metadata(state, 'initial_value', state.value)
        else:
            if isinstance(initial_value, Variable):
                try:
                    initial_value = check_variable_shape_compatibility(initial_value, state)
                except Exception as e:
                    raise ValueError(f"Error with initial value argument. {e}")
            else:
                try:
                    initial_value = ingest_value(initial_value)
                except Exception as e:
                    raise ValueError(f"Error with initial value. {e}")
            self.add_state_metadata(state, 'initial_value', initial_value)

        # Check if user provided a tolerance
        self.add_tolerance(state, tolerance)

    def _inline_update_states(self):
        # while not converged:
        #    x0_new = x0_state_update(x0_old, x1_old, ... xn_old)
        #    x1_new = x1_state_update(x0_new, x1_old, ... xn_old)
        #    ...
        #    xn_new = xn_state_update(x0_new, x1_new, ... xn_old)
        for current_state, current_residual in self.state_to_residual_map.items():
            # compute residuals
            self.update_residual()

            # get current state value and residual value
            current_state_value = current_state.value
            current_residual_value = current_residual.value

            # update current state value
            if self.state_metadata[current_state]['state_update'] is None:
                current_state.value = current_state_value - current_residual_value
            else:
                current_state.value = self.state_metadata[current_state]['state_update'].value
