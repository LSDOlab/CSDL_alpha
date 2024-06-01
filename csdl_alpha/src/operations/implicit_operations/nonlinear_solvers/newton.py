from csdl_alpha.src.operations.implicit_operations.nonlinear_solvers.nonlinear_solver import NonlinearSolver, check_variable_shape_compatibility, check_run_time_tolerance
from csdl_alpha.src.operations.implicit_operations.nonlinear_solvers.fixed_point import FixedPoint
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import scalarize, ingest_value
import numpy as np

from typing import Union

class Newton(FixedPoint):
    
    def __init__(
            self,
            name = 'newton_nlsolver',
            print_status = True,
            tolerance=1e-10,
            max_iter=100,
            elementwise_states=False,
        ):
        """
        Newton method solver
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


    def _preprocess_run(self):
        """
        Preprocess the solver before running
        """
        full_residual_jacobian = self.get_full_residual_jacobian()
        self.add_intersection_target(full_residual_jacobian)
        self.add_intersection_source(full_residual_jacobian)

    def _inline_update_states(self):
        # get residuals
        res_val = {}
        residual_vector = np.zeros((self.total_state_size,))
        for current_state, current_residual in self.state_to_residual_map.items():
            # get current state value and residual value
            current_residual_value = current_residual.value

            il = self.state_metadata[current_state]['index_lower']
            iu = self.state_metadata[current_state]['index_upper']
            residual_vector[il:iu] = current_residual_value.flatten()

        # Solve residual Jacobian system
        solved_system = np.linalg.solve(self.full_residual_jacobian.value, residual_vector)

        # update states
        for current_state, current_residual in self.state_to_residual_map.items():
            il = self.state_metadata[current_state]['index_lower']
            iu = self.state_metadata[current_state]['index_upper']
            current_state.value = current_state.value - solved_system[il:iu].reshape(current_state.shape)
