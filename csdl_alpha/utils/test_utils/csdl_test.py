import numpy as np

class CSDLTest():

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
        ):
        import csdl_alpha as csdl
        import numpy as np
        recorder = csdl.get_current_recorder()
        recorder.stop()

        if compare_values is None:
            compare_values = []
        if compare_derivatives is None:
            compare_derivatives = []

        from numpy.testing import assert_array_equal

        # TODO: make compatible with sparse variables
        for ind, testing_pair in enumerate(compare_values):
            testing_pair.compare(ind+1)

        # for variable, real_value in compare_values.items():
        #     if not isinstance(variable, csdl.Variable):
        #         raise ValueError(get_testing_error_string(f"compare_values key {variable} is not a csdl.Variable"))
        #     if not isinstance(real_value, (np.ndarray)):
        #         raise ValueError(get_testing_error_string(f"compare_values value {real_value} is not a numpy array"))
            
        #     # assert shapes:
        #     assert variable.shape == real_value.shape

        #     # assert values:
        #     assertion_error_str = get_assertion_error_string(variable, real_value)
        #     assert_array_equal(variable.value, real_value, err_msg=assertion_error_str)

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
        return error_str

# class TestingPairs():

#     def __init__(self):
#         self.pairs = []

#     def add_pair(self, checking_pair:CheckPair):
#         self.pairs.append(checking_pair)

#     def compare(self):
#         for i, pair in enumerate(self.pairs):
#             pair.compare(index = i)
