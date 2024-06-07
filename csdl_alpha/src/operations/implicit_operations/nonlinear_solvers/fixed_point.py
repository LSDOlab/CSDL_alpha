from csdl_alpha.src.operations.implicit_operations.nonlinear_solvers.nonlinear_solver import NonlinearSolver, check_variable_shape_compatibility, check_run_time_tolerance
from csdl_alpha.src.graph.variable import Variable
from csdl_alpha.utils.inputs import scalarize, ingest_value
import numpy as np

from typing import Union

class FixedPoint(NonlinearSolver):
    
    def __init__(
            self,
            name,
            print_status,
            tolerance,
            max_iter,
            elementwise_states,
            residual_jac_kwargs,
        ):
        """
        Base class for standard fixed point iteration solvers. Not to be used directly.
        """
        # well now this just seems redundant
        super().__init__(
            name = name,
            print_status = print_status,
            tolerance = tolerance,
            max_iter = max_iter,
            elementwise_states = elementwise_states,
            residual_jac_kwargs = residual_jac_kwargs,
        )

    def add_state(*args, **kwargs):
        """
        Not implemented
        """
        raise NotImplementedError("add_state method not implemented for this solver")
    
    def _inline_update_states(self):
        """
        Not implemented
        """
        raise NotImplementedError("_inline_update_states method not implemented for this solver")

    def _inline_solve_(self):
        """
        Solves the implicit operation graph using Nonlinear Gauss-Seidel
        """

        iter = 0

        self._inline_set_initial_values()

        while True:
            # update all states
            self._inline_update_states()
            
            # update residuals to check
            self.update_residual()

            # check convergence
            converged = self._inline_check_converged()

            # if solved or maxiter, end loop
            if converged:
                break
            iter += 1
            if iter >= self.metadata['max_iter']:
                break
        # print status
        if self.print_status:
            print(self._inline_print_nl_status(iter, converged))