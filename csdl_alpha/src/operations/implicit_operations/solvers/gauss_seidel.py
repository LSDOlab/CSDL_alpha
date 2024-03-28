from csdl_alpha.src.operations.implicit_operations.solvers.nonlinear_solver import NonlinearSolver
from csdl_alpha.src.graph.variable import Variable
from typing import Union

class GaussSeidel(NonlinearSolver):
    
    def __init__(self, name = 'gs_nlsolver', tolerance=1e-6, max_iter=1000):
        """
        A Gauss-Seidel solver!
        """

        super().__init__(name = name)

        self.metadata['tolerance'] = tolerance
        self.metadata['max_iter'] = max_iter

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
        self.state_metadata[state] = {
            'state_update': state_update,
        }
        if isinstance(state_update, Variable):
            self.add_intersection_target(state_update)

        # Check if user provided an initial value
        if initial_value is None:
            self.state_metadata[state]['initial_value'] = state.value
        else:
            self.state_metadata[state]['initial_value'] = initial_value

        # Check if user provided a tolerance
        if tolerance is None:
            self.state_metadata[state]['tolerance'] = self.metadata['tolerance']
        else:
            self.state_metadata[state]['tolerance'] = tolerance


