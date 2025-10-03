import csdl_alpha.utils.testing_utils as csdl_tests

def f(x, i:int):
    import csdl_alpha as csdl
    if i > 0: x = csdl.sin(x)
    if i > 1: x = csdl.cos(x)
    if i > 2: x = csdl.tan(x)
    if i > 3: x = csdl.exp(x)
    if i > 4: x = csdl.log(x)
    if i > 5: x = csdl.arctan(x)
    if i > 6: x = 2.0-x-5.0
    if i > 7: x = 1.1/x/2.0
    if i > 8: x = 1.1*x*3.0
    if i > 9: x = 0.1+x+4.0
    return x

class TestInvert(csdl_tests.CSDLTest):
    def test_invert_simple(self):
        self.prep()
        import csdl_alpha as csdl
        import numpy as np

        for i in range(11):
            recorder = csdl.Recorder(inline = True, debug=True)
            recorder.start()

            x_value = np.array([[0.5]])
            x = csdl.Variable(value=x_value, name = 'x')
            y = f(x,i)
            InversionTransform = csdl.transforms.EqualityInversion()
            inverted_lhs, inverted_rhs, aux = InversionTransform.transform(lhs=y, rhs=y, aux_info=True, debug=False, target=x)
            print(f'Checking inversion of {i} operations: {[op.name for op in aux]} ...')

            assert np.allclose(inverted_lhs.value, inverted_rhs.value), f'Inversion failed at {i} operations: {inverted_lhs.value} != {inverted_rhs.value}'
            assert np.allclose(inverted_lhs.value, x_value), f'Inversion failed at {i} operations: {inverted_lhs.value} != {x_value}'

    def test_invert_simple_2(self):
        self.prep()
        import csdl_alpha as csdl
        import numpy as np

        for i in range(11):
            recorder = csdl.Recorder(inline = True, debug=True)
            recorder.start()

            x_value = np.array([0.5])
            x = csdl.Variable(value=x_value, name = 'x')
            y = f(x,i)
            InversionTransform = csdl.transforms.EqualityInversion()
            inverted_lhs, inverted_rhs, aux = InversionTransform.transform(lhs=y, rhs=y, aux_info=True, debug=False, target=x)
            print(f'Checking inversion of {i} operations: {[op.name for op in aux]} ...')

            assert np.allclose(inverted_lhs.value, inverted_rhs.value), f'Inversion failed at {i} operations: {inverted_lhs.value} != {inverted_rhs.value}'
            assert np.allclose(inverted_lhs.value, x_value), f'Inversion failed at {i} operations: {inverted_lhs.value} != {x_value}'

    def test_invert_simple_3(self):
        self.prep(always_build_inline=True)
        import csdl_alpha as csdl
        import numpy as np

        for i in range(11):
            recorder = csdl.Recorder(inline = True, debug=True)
            recorder.start()

            x_value = np.array([0.5, 0.6])
            x = csdl.Variable(value=x_value, name = 'x')
            y = f(x,i).T()
            InversionTransform = csdl.transforms.EqualityInversion()
            inverted_lhs, inverted_rhs, aux = InversionTransform.transform(lhs=y, rhs=y, aux_info=True, debug=False, target=x)
            print(f'Checking inversion of {i} operations: {[op.name for op in aux]} ...')

            assert np.allclose(inverted_lhs.value, inverted_rhs.value), f'Inversion failed at {i} operations: {inverted_lhs.value} != {inverted_rhs.value}'
            assert np.allclose(inverted_lhs.value, x_value), f'Inversion failed at {i} operations: {inverted_lhs.value} != {x_value}'


    def test_invert_target(self):
        self.prep(always_build_inline=True)
        import csdl_alpha as csdl
        import numpy as np

        compare_values = []

        x_value, e_value = np.array([[0.5]]), np.e
        x = csdl.Variable(value=x_value, name = 'x')
        e = csdl.Variable(value=e_value, name = 'e')
        y = csdl.power(e, x)

        # Invert wrt target x
        InversionTransform = csdl.transforms.EqualityInversion()
        inverted_lhs, inverted_rhs, aux = InversionTransform.transform(lhs=y, rhs=y, aux_info=True, target=x, debug=False)
        print(f'Checking inversion: {[op.name for op in aux]} ...')
        assert np.allclose(inverted_lhs.value, inverted_rhs.value), f'{inverted_lhs.value} != {inverted_rhs.value}'
        assert np.allclose(inverted_lhs.value, x_value), f'{inverted_lhs.value} != {x_value}'
        compare_values += [csdl_tests.TestingPair(inverted_lhs, x_value, tag = 'test_1')]

        # invert wrt target e
        InversionTransform = csdl.transforms.EqualityInversion()
        inverted_lhs, inverted_rhs, aux = InversionTransform.transform(lhs=y, rhs=y, aux_info=True, target=e, debug=False)
        print(f'Checking inversion: {[op.name for op in aux]} ...')
        assert np.allclose(inverted_lhs.value, inverted_rhs.value), f'{inverted_lhs.value} != {inverted_rhs.value}'
        assert np.allclose(inverted_lhs.value, e_value), f'{inverted_lhs.value} != {e_value}'
        compare_values += [csdl_tests.TestingPair(inverted_lhs, (np.array(e_value)).reshape(inverted_lhs.shape), tag = 'test_2')]

        # invert wrt untargetable variable
        x_value, e_value = np.array([[0.5], [0.1]]), np.e
        x = csdl.Variable(value=x_value, name = 'x')
        e = csdl.Variable(value=e_value, name = 'e')
        y = csdl.power(e, x)*2.0
        # Can't invert wrt e because its a scalar and y is not
        InversionTransform = csdl.transforms.EqualityInversion()
        inverted_lhs, inverted_rhs, aux = InversionTransform.transform(lhs=y, rhs=y, aux_info=True, target=e, debug=False)
        print(f'Checking inversion: {[op.name for op in aux]} ...')
        assert np.allclose(inverted_lhs.value, inverted_rhs.value), f'{inverted_lhs.value} != {inverted_rhs.value}'

        # Invert naturally w/o target
        InversionTransform = csdl.transforms.EqualityInversion()
        inverted_lhs, inverted_rhs, aux = InversionTransform.transform(lhs=y, rhs=y, aux_info=True, debug=False)
        print(f'Checking inversion: {[op.name for op in aux]} ...')
        assert np.allclose(inverted_lhs.value, inverted_rhs.value), f'{inverted_lhs.value} != {inverted_rhs.value}'
        assert np.allclose(inverted_lhs.value, x_value), f'{inverted_lhs.value} != {x_value}'
        compare_values += [csdl_tests.TestingPair(inverted_lhs, x_value, tag = 'test_3')]

        self.run_tests(compare_values = compare_values, verify_derivatives=True)


if __name__ == '__main__':
    test = TestInvert()
    test.overwrite_backend = 'inline'
    test.test_invert_target()
    # test.test_invert_simple()
    # test.test_invert_simple_2()
    # test.test_invert_simple_3()