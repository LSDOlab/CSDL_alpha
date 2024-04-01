class CSDLTest():

    def prep(self, inline = True, **kwargs):
        """
        preprocessing for running tests to check csdl computations.

        kwargs are passed to the recorder constructor.
        """
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
            compare_values = {}
        if compare_derivatives is None:
            compare_derivatives = {}

        from numpy.testing import assert_array_equal

        # TODO: make compatible with sparse variables
        for variable, real_value in compare_values.items():
            if not isinstance(variable, csdl.Variable):
                raise ValueError(get_testing_error_string(f"compare_values key {variable} is not a csdl.Variable"))
            if not isinstance(real_value, (np.ndarray)):
                raise ValueError(get_testing_error_string(f"compare_values value {real_value} is not a numpy array"))
            
            # assert shapes:
            assert variable.shape == real_value.shape

            # assert values:
            assertion_error_str = get_assertion_error_string(variable, real_value)
            assert_array_equal(variable.value, real_value, err_msg=assertion_error_str)

def get_testing_error_string(error_string):
    return f"Test implementation error: {error_string}"

def get_assertion_error_string(csdl_variable, real_value):
    return f"Value of csdl_variable (name={csdl_variable.name}) is \n{csdl_variable.value} \nbut should be \n{real_value}"
