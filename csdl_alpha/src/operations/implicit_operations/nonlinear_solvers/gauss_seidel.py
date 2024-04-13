from csdl_alpha.src.operations.implicit_operations.nonlinear_solvers.nonlinear_solver import NonlinearSolver
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import scalarize, ingest_value
import numpy as np

from typing import Union

class GaussSeidel(NonlinearSolver):
    
    def __init__(
            self,
            name = 'gs_nlsolver',
            print_status = True,
            tolerance=1e-10,
            max_iter=100,
        ):
        """
        A Gauss-Seidel solver
        """
        # well now this just seems redundant
        super().__init__(
            name = name,
            print_status = print_status,
            tolerance = tolerance,
            max_iter = max_iter
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
        self.add_state_metadata(state, 'state_update', state_update, is_input=False)

        if isinstance(state_update, Variable):
            self.add_intersection_target(state_update)

        # Check if user provided an initial value
        if initial_value is None:
            self.add_state_metadata(state, 'initial_value', state.value)
        else:
            self.add_state_metadata(state, 'initial_value', initial_value)

        # Check if user provided a tolerance
        if tolerance is None:
            self.add_state_metadata(state, 'tolerance', self.metadata['tolerance'])
        else:
            if isinstance(tolerance, Variable):
                if tolerance.value.size != 1:
                    raise ValueError(f"Tolerance must be a scalar. {tolerance.shape} given")
            else:
                tolerance = scalarize(tolerance)
            self.add_state_metadata(state, 'tolerance', tolerance)


    def _inline_solve_(self):
        """
        Solves the implicit operation graph using Nonlinear Gauss-Seidel
        """

        # perform NLBGS iterations:
        # while not converged:
        #    x0_new = x0_old - r0(x0_old, x1_old, ... xn_old)
        #    x1_new = x1_old - r1(x0_new, x1_old, ... xn_old)
        #    ...
        #    xn_new = xn_old - rn(x0_new, x1_new, ... xn_old)

        iter = 0

        self._inline_set_initial_values()

        while True:
            # loop through all residuals
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
            # update residuals to check
            self.update_residual()

            # check convergence
            converged = True
            for current_state, current_residual in self.state_to_residual_map.items():

                # get current residual and error
                current_residual_value = current_residual.value
                error = np.linalg.norm(current_residual_value.flatten())
                # print(f'iteration {iter}, {current_residual} error: {error}')

                # if any of the residuals do not meet tolerance, no need to compute errors for other residuals
                tol = self.state_metadata[current_state]['tolerance']
                if isinstance(tol, Variable):
                    tol = tol.value

                if np.isnan(error):
                    raise ValueError(f'Residual is NaN for {current_residual.name}')
                if error > tol:
                    converged = False
                    break

            # if solved or maxiter, end loop
            if converged:
                break
            iter += 1
            if iter >= self.metadata['max_iter']:
                break
        # print status
        if self.print_status:
            print(self._inline_print_nl_status(iter, converged))