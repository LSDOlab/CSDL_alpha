import numpy as np
import pytest

# @pytest.fixture(scope="class")
# def pytest_req(request):
#     request.cls.poo = request.config.getoption("--custom-option")

# @pytest.mark.usefixtures("poo")
class CSDLTest():

    @pytest.fixture(autouse=True)
    def inject_config(self, request):
        self._config = request.config

    def prep(self, inline = True, **kwargs):
        """
        preprocessing for running tests to check csdl computations.

        kwargs are passed to the recorder constructor.
        """
        from csdl_alpha.utils.hard_reload import hard_reload
        hard_reload()

        kwargs['inline'] = inline
        import csdl_alpha as csdl
        recorder = csdl.build_new_recorder(**kwargs)
        recorder.start()

    def run_tests(
            self,
            compare_values = None,
            compare_derivatives = None,
            verify_derivatives = False,
            ignore_derivative_fd_error:set = None,
            step_size = 1e-6,
            ignore_constants = False,
            turn_off_recorder = True,
        ):
        self.backend_type = self._config.getoption("--backend")
        if self.backend_type not in ['inline', 'jax']:
            raise ValueError(f"Backend type {self.backend_type} not supported for testing. Use 'inline', 'jax'")
        
        import csdl_alpha as csdl
        import numpy as np
        recorder = csdl.get_current_recorder()
        if turn_off_recorder:
            recorder.stop()

        # TEST THREE THINGS:
        # 1. Compare forward evaluation values according to using testing_pairs
        # 2. Compare derivatives with finite difference if needed
        # 3. Rerun the graph to make sure we can rerun

        # Find all inputs and outputs
        graph_insights = recorder.gather_insights()
        wrts_all = list(graph_insights['input_nodes'])
        all_inputs = []
        for i, wrt in enumerate(wrts_all):
            from csdl_alpha.src.graph.variable import Constant
            if ignore_constants and isinstance(wrt, Constant):
                continue
            all_inputs.append(wrt)
        all_outputs = [testing_pair.csdl_variable for testing_pair in compare_values]
        
        # 1: 
        self.check_values(
            compare_values,
            recorder,
            all_inputs,
            all_outputs,
        )

        # 2:
        if verify_derivatives:
            self.compare_derivatives(
                recorder,
                all_inputs,
                compare_values,
                ignore_derivative_fd_error,
                step_size,
                ignore_constants,
            )

        # 3.
        self.rerun(recorder)

    def check_values(
            self,
            compare_values,
            recorder,
            all_inputs,
            all_outputs,
            ):
        if self.backend_type == 'jax':
            from csdl_alpha.backends.jax.graph_to_jax import create_jax_function
            jax_func = create_jax_function(
                graph = recorder.active_graph,
                outputs = all_outputs,
                inputs = all_inputs,
            )
            import jax
            import jax.numpy as jnp
            jax_func = jax.jit(jax_func)
            jnp_outputs = jax_func(*[jnp.array(input.value) for input in all_inputs])
            
            for ind, testing_pair in enumerate(compare_values):
                testing_pair.csdl_variable.value = 0.0
                testing_pair.csdl_variable.value = np.array(jnp_outputs[ind])
        for ind, testing_pair in enumerate(compare_values):
            testing_pair.compare(ind+1)

    def compare_derivatives(
            self,
            recorder,
            all_inputs,
            compare_values,
            ignore_derivative_fd_error,
            step_size,
            ignore_constants,
        ):
        if self.backend_type == 'inline':
            if ignore_derivative_fd_error is None:
                ignore_derivative_fd_error = set()

            recorder.start()
            # from csdl_alpha.src.operations.derivatives.utils import verify_derivatives_inline
            
            wrts = all_inputs
            ofs = [testing_pair.csdl_variable for testing_pair in compare_values]
            of_wrt_meta_data = {}
            for testing_pair in compare_values:
                tag = testing_pair.tag
                if tag is None:
                    tag = ''
                rel_error = 10**(-testing_pair.decimal)
                for wrt in wrts:
                    if (wrt in ignore_derivative_fd_error) or (testing_pair.csdl_variable in ignore_derivative_fd_error):
                        rel_error = 2.0
                    of_wrt_meta_data[(testing_pair.csdl_variable, wrt)] = {
                        'tag': tag,
                        'max_rel_error': rel_error,   
                    }
            
            # verify_derivatives_inline(ofs, wrts, step_size, of_wrt_meta_data = of_wrt_meta_data)

            import csdl_alpha as csdl
            csdl.derivative_utils.verify_derivatives(
                ofs,
                wrts,
                step_size,
                verification_options=of_wrt_meta_data,
            )

            recorder.stop()
        else:
            raise ValueError(f"no trigger")
        # exit('END 2')

    def rerun(self, recorder):
        if self.backend_type == 'inline':
            recorder.execute()
        else:
            raise ValueError(f"no trigger")

    def docstest(self, obj):
        # self.prep()
        # import doctest
        # import csdl_alpha as csdl
        # import numpy as np
        # doctest.run_docstring_examples(
        #     obj,
        #     globs = {'csdl': csdl, 'np': np},
        #     optionflags=doctest.FAIL_FAST,
        # )
        self.prep()
        import doctest
        import csdl_alpha as csdl
        import numpy as np

        # Create a new instance of a DocTestFinder and DocTestRunner class
        finder = doctest.DocTestFinder()
        runner = doctest.DocTestRunner(verbose=False)

        # Find the tests
        tests = finder.find(obj, globs={'csdl': csdl, 'np': np})

        # Run the tests
        for test in tests:
            runner.run(test)

        # If there were any failures, raise an exception
        if runner.failures > 0:
            raise Exception(f"{runner.failures} doctest(s) failed")


    # def run_tests_old(
    #         self,
    #         compare_values = None,
    #         compare_derivatives = None,
    #         verify_derivatives = False,
    #         ignore_derivative_fd_error:set = None,
    #         step_size = 1e-6,
    #         ignore_constants = False,
    #         turn_off_recorder = True,
    #     ):

    #     backend_type = self._config.getoption("--backend")
    #     if backend_type not in ['inline', 'jax_sim', 'py_sim']:
    #         raise ValueError(f"Backend type {backend_type} not supported for testing")

    #     import csdl_alpha as csdl
    #     import numpy as np
    #     recorder = csdl.get_current_recorder()
    #     if turn_off_recorder:
    #         recorder.stop()

    #     if compare_values is None:
    #         compare_values = []
    #     if compare_derivatives is None:
    #         compare_derivatives = []

    #     from numpy.testing import assert_array_equal

    #     # TODO: make compatible with sparse variables
    #     for ind, testing_pair in enumerate(compare_values):
    #         testing_pair.compare(ind+1)
        
    #     # Add derivatives to the graph and verify with finite difference if needed. Recorder needs to be re-started
    #     if verify_derivatives:
    #         if ignore_derivative_fd_error is None:
    #             ignore_derivative_fd_error = set()

    #         recorder.start()
    #         from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
            
    #         graph_insights = recorder.gather_insights()
    #         wrts_all = list(graph_insights['input_nodes'])
    #         wrts = []
    #         j = 0
    #         for i, wrt in enumerate(wrts_all):
    #             from csdl_alpha.src.graph.variable import Constant
    #             # TODO: Find a deterministic way to skip certain constants
    #             # if isinstance(wrt, Constant):
    #             #     # check if i is even:
    #             #     j+=1
    #             #     if (j) % 3 == 0:
    #             #         continue
    #             if ignore_constants and isinstance(wrt, Constant):
    #                 continue
    #             wrts.append(wrt)
    #         ofs = [testing_pair.csdl_variable for testing_pair in compare_values]
    #         of_wrt_meta_data = {}
    #         for testing_pair in compare_values:
    #             tag = testing_pair.tag
    #             if tag is None:
    #                 tag = ''
    #             rel_error = 10**(-testing_pair.decimal)
    #             for wrt in wrts:
    #                 if (wrt in ignore_derivative_fd_error) or (testing_pair.csdl_variable in ignore_derivative_fd_error):
    #                     rel_error = 2.0
    #                 of_wrt_meta_data[(testing_pair.csdl_variable, wrt)] = {
    #                     'tag': tag,
    #                     'max_rel_error': rel_error,   
    #                 }
            
    #         verify_derivatives_inline(ofs, wrts, step_size, of_wrt_meta_data = of_wrt_meta_data)

    #         recorder.stop()

    #     # run the graph again to make sure it actually runs twice.
    #     recorder.execute()
    #     # recorder.print_graph_structure()
    #     # recorder.visualize_graph()


def get_testing_error_string(error_string):
    return f"Test implementation error: {error_string}"

class TestingPair():
    from csdl_alpha.src.graph.variable import Variable

    def __init__(self, 
            csdl_variable:Variable,
            real_value:np.ndarray,
            decimal = 11,
            tag:str = None,
        ):
        """
        Class to compare a csdl variable with a real value for unit tests.

        Args:
            csdl_variable (Variable): csdl variable to compare
            real_value (np.ndarray): real value to compare
            decimal (int): decimal places for tolerance
            tag (str): tag for the pair
        """
        from csdl_alpha.src.graph.variable import Variable
        if not isinstance(csdl_variable, Variable):
            raise TypeError(get_testing_error_string(f"compare_values key {csdl_variable} is not a csdl.Variable"))
        if not isinstance(real_value, (np.ndarray)):
            raise TypeError(get_testing_error_string(f"compare_values value {real_value} is not a numpy array"))

        self.csdl_variable = csdl_variable
        self.real_value = real_value
        self.decimal = decimal

        if tag is None:
            self.tag = csdl_variable.name
        else:
            self.tag = tag

        # TODO: If inline, compare values as tesing pair is created?
        # if self.csdl_variable is not None:
        #     self.compare('inline')

    def compare(self, ind):
        """
        Tests the shapes and values of the csdl variable and the real value.
        """
        if self.csdl_variable.shape != self.real_value.shape:
            raise AssertionError(self.get_assertion_error_string(ind, 'shape'))

        from numpy.testing import assert_array_almost_equal
        assert_array_almost_equal(
            self.csdl_variable.value,
            self.real_value,
            decimal = self.decimal,
            err_msg = self.get_assertion_error_string(ind, 'value')
        )

    def get_assertion_error_string(self, ind, error_type):
        error_str = "\n"
        error_str += f"{error_type} assertion error in\n"
        error_str += f"Var/value pair:   \"{self.tag}\"\n"
        error_str += f"Index in list:    {ind}\n"
        error_str += f"Variable name:    {self.csdl_variable.name}\n"

        if error_type == 'value':
            error_str += f"\nVariable value:   \n{self.csdl_variable.value}\n\n"
            error_str += f"Real value:       \n{self.real_value}\n"
        elif error_type == 'shape':
            error_str += f"\nVariable shape:   {self.csdl_variable.shape}\n"
            error_str += f"Real shape:       {self.real_value.shape}\n"
        return error_str

# class TestingPairs():

#     def __init__(self):
#         self.pairs = []

#     def add_pair(self, checking_pair:CheckPair):
#         self.pairs.append(checking_pair)

#     def compare(self):
#         for i, pair in enumerate(self.pairs):
#             pair.compare(index = i)
