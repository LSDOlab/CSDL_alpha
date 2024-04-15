'''
Paraboloid:
'''
if __name__ == '__main__':
    import csdl_alpha as csdl
    import numpy as np

    class ParaboloidModel:

        def __init__(self, a: float, b: float, c: float):
            """
            Initializes a Paraboloid object with the given coefficients.

            Parameters
            ----------
            a : float
                The coefficient of the x^2 term in the paraboloid equation.
            b : float
                The coefficient of the y^2 term in the paraboloid equation.
            c : float
                The coefficient of the xy term in the paraboloid equation.
            """
            self.a = a
            self.b = b
            self.c = c

        def evaluate(self, x: csdl.Variable, y: csdl.Variable, name: str = 'paraboloid'):
            """
            Evaluate the paraboloid function at the given values of x and y.

            Parameters
            ----------
            x : csdl.Variable
                The value of x.
            y : csdl.Variable
                The value of y.

            Returns
            -------
            csdl.Variable
                The result of evaluating the paraboloid function at the given values of x and y.
            """
            with csdl.Namespace(name):
                f = csdl.square(x - self.a) + x * y + csdl.square(y + self.b) - self.c
                f.add_name('f')  # This gives f the name '...paraboloid.f'
            return f

    recorder = csdl.Recorder(inline=True, debug=True, auto_hierarchy=True)
    recorder.start()


    csdl.enter_namespace('test')
    x = csdl.Variable(value=2., name='x')
    csdl.exit_namespace()
    x.save()
    x.add_tag('banana')
    x.add_tag('apple')
    y = csdl.Variable(value=1., name='y')

    a = 12
    b = 5
    c = 30
    # Add a sub-model to the current model
    paraboloid_model = ParaboloidModel(a, b, c)

    # Call the evaluate method of the model functionally.
    f = paraboloid_model.evaluate(x, y)
    f.save()

    recorder.active_graph.visualize()

    assert f.value == np.array([108])
    # f.print_trace()
    csdl.save_all_variables()
    csdl.inline_save('test')
    # variables = csdl.import_h5py('test.hdf5', 'inline')
    

    recorder.stop()