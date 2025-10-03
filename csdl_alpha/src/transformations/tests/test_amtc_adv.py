import csdl_alpha as csdl
import numpy as np
import pytest

def simple_func(x1,x2,x3, a):
    return x1**a+x2**2+x3

def f10(x):
    return x**2.0

def f01(y):
    return csdl.sin(y+0.5)**2.0

def f11(x, y):
    return csdl.average(csdl.outer(x,y))

def test_amtc_simple():
    recorder = csdl.Recorder(inline = False, debug=True)
    recorder.start()
    x1 = csdl.Variable(value=np.array([1.0]), name = 'x1')
    x2 = csdl.Variable(value=np.array([1.0]), name = 'x2')
    x3 = csdl.Variable(value=np.array([1.0]), name = 'x3')
    a = csdl.Variable(value=np.array([3.0]), name = 'a')
    z = simple_func(x1, x2, x3, a)

    # amtc:
    AMTC = csdl.transforms.AMTC()
    rv_dict = {
        x1: csdl.Variable(value=np.array([[1.0], [2.0], [3.0]])),
        x2: csdl.Variable(value=np.array([[2.0], [3.0], [4.0]])),
        x3: csdl.Variable(value=np.array([[3.0], [4.0], [5.0]])),
    }
    tensor_grid_evalations = AMTC.transform(
        rvs = rv_dict,
        outputs = [z]
    )
    avg = csdl.average(tensor_grid_evalations[z])

    # derivatives:
    # avg_deriv = csdl.derivative(avg, [rv_dict[x1], rv_dict[x2], rv_dict[x3], a])
    recorder.execute()

    # check values for testing
    real_tensor_eval = np.array([8, 9, 10, 13, 14, 15, 20, 21, 22, 15, 16, 17, 20, 21, 22, 27, 28, 29, 34, 35, 36, 39, 40, 41, 46, 47, 48]).reshape(27,1)
    np.testing.assert_allclose(real_tensor_eval, tensor_grid_evalations[z].value)
    np.testing.assert_allclose(avg.value, np.array([25.666666666]))

    # Check derivatives
    from csdl_alpha import derivative_utils
    derivative_utils.verify_derivatives(
        ofs = avg,
        wrts = [rv_dict[x1], rv_dict[x2], rv_dict[x3], a],
        step_size=1e-6,
    )
    a.value = 4
    derivative_utils.verify_derivatives(
        ofs = avg,
        wrts = [rv_dict[x1], rv_dict[x2], rv_dict[x3], a],
        step_size=1e-6,
    )

def test_amtc_simple2():
    recorder = csdl.Recorder(inline = False, debug=True)
    recorder.start()
    x1 = csdl.Variable(value=np.array([1.0,1.0]), name = 'x1')
    x2 = csdl.Variable(value=np.array([1.0,1.0]), name = 'x2')
    x3 = csdl.Variable(value=np.array([1.0,1.0]), name = 'x3')
    a = csdl.Variable(value=np.array([3.0,4.0]), name = 'a')
    z = simple_func(x1, x2, x3, a)

    # amtc:
    AMTC = csdl.transforms.AMTC()
    rv_dict = {
        x1: csdl.Variable(value=np.array([[1.0,1.5], [2.0,2.5], [3.0,3.5]])),
        # x2: csdl.Variable(value=np.array([[2.0,2.5], [3.0,3.5], [4.0,4.5]])),
        # x3: csdl.Variable(value=np.array([[3.0,3.5], [4.0,4.5], [5.0,5.5]])),
    }
    tensor_grid_evalations = AMTC.transform(
        rvs = rv_dict,
        outputs = [z]
    )
    avg = csdl.average(tensor_grid_evalations[z])

    inputs = list(rv_dict.values()) + [a]
    outputs = list(tensor_grid_evalations.values()) + [avg]

    # Try JAX sim
    run_jax_sim(inputs, outputs, recorder)

    # derivatives:
    avg_deriv = csdl.derivative(avg, [rv_dict[x1], a])
    recorder.execute()

def test_amtc_transform():
    
    recorder = csdl.Recorder(inline = False, debug=True)
    recorder.start()
    a = csdl.Variable(value=np.array([3.0]), name = 'a')
    x_0 = csdl.Variable(value=np.array([1.0]), name = 'x_0')
    x_1 = csdl.Variable(value=np.array([1.0]), name = 'x_1')
    x = x_0+x_1*2.0
    y = csdl.Variable(value=np.array([1.0]), name = 'y')

    x_out = f10(x)
    y_out = f01(y)
    z = f11(x_out, y_out)*a
    z_fake = x_out**3.0+2.0

    # Setup:
    AMTC = csdl.transforms.AMTC()
    rv_dict = {
        x_0: csdl.Variable(value=np.array([[1.0], [2.0]])),
        x_1: csdl.Variable(value=np.array([[1.0], [2.0]])),
        y: csdl.Variable(value=np.array([[3.0], [4.0]])),
    }
    tensor_grid_evalations = AMTC.transform(
        rvs = rv_dict,
        outputs = [z]
    )
    # print(tensor_grid_evalations[z].value)

    inputs = list(rv_dict.values()) + [a]
    outputs = list(tensor_grid_evalations.values())

    # Try JAX sim
    run_jax_sim(inputs, outputs, recorder)

    # deriv = csdl.derivative(tensor_grid_evalations[z], rv_dict[x_0])
    # recorder.execute()
    # # print(deriv.value)

def test_amtc_transform_tensor_1rv():
    recorder = csdl.Recorder(inline = False, debug=False)
    recorder.start()    
    x_0 = csdl.Variable(value=np.array([[1.0], [1.0]]), name = 'x_0')
    x_1 = csdl.Variable(value=np.array([[1.0], [1.0]]), name = 'x_1')
    x = x_0+x_1*2.0
    y = csdl.Variable(value=np.array([[1.0], [1.0]]), name = 'y')

    x_out = f10(x)
    y_out = f01(y)
    z = f11(x_out, y_out)
    z_fake = x_out**3.0+2.0

    # Setup:
    # with pytest.raises(ValueError):
    AMTC = csdl.transforms.AMTC()
    rv_dict = {
        x_0: csdl.Variable(value=np.array([[[1.0], [2.0]],[[3.0], [4.0]]])),
        # x_1: csdl.Variable(value=np.array([[1.0], [2.0]])),
        # y: csdl.Variable(value=np.array([[3.0], [4.0]])),
    }
    tensor_grid_evalations = AMTC.transform(
        rvs = rv_dict,
        outputs = [z, y_out],
    )

    inputs = list(rv_dict.values()) + [x_1]
    outputs = list(tensor_grid_evalations.values())

    # Try JAX sim
    run_jax_sim(inputs, outputs, recorder)

    recorder.execute()
    # print(tensor_grid_evalations[z].value)
    # print(tensor_grid_evalations[y_out].value)


def test_amtc_transform_tensor_2rv():
    recorder = csdl.Recorder(inline = False, debug=False)
    recorder.start()    
    x_0 = csdl.Variable(value=np.array([[1.0], [1.0]]), name = 'x_0')
    x_1 = csdl.Variable(value=np.array([[1.0], [1.0]]), name = 'x_1')
    x = x_0+x_1*2.0
    y = csdl.Variable(value=np.array([[1.0], [1.0]]), name = 'y')

    x_out = f10(x)
    y_out = f01(y)
    z = f11(x_out, y_out)
    z_fake = x_out**3.0+2.0

    # Setup:
    # with pytest.raises(ValueError):
    AMTC = csdl.transforms.AMTC()
    rv_dict = {
        x_0: csdl.Variable(value=np.array([[[1.0], [2.0]],[[3.0], [4.0]]])),
        # x_1: csdl.Variable(value=np.array([[1.0], [2.0]])),
        y: csdl.Variable(value=np.array([[[5.0], [6.0]],[[7.0], [8.0]],[[7.0], [8.0]]])),
    }
    tensor_grid_evalations = AMTC.transform(
        rvs = rv_dict,
        outputs = [z, y_out],
    )
    inputs = list(rv_dict.values()) + [x_1]
    outputs = list(tensor_grid_evalations.values())

    # Try JAX sim
    run_jax_sim(inputs, outputs, recorder)


# def test_amtc_transform_tensor_2rv_3():
#     recorder = csdl.Recorder(inline = False, debug=False)
#     recorder.start()
#     a = csdl.Variable(value=np.array([3.0]), name = 'a')
#     x_0 = csdl.Variable(value=np.array([[1.0], [1.0]]), name = 'x_0')
#     x_1 = csdl.Variable(value=np.array([[1.0], [1.0]]), name = 'x_1')
#     x = x_0+x_1*2.0
#     y = csdl.Variable(value=np.array([[1.0], [1.0]]), name = 'y')

#     x_out = f10(x*x_1)
#     y_out = f01(y*x_1)
#     z = f11(x_out, y_out)
#     z_fake = (a*z)**3.0+2.0

#     # Setup:
#     # with pytest.raises(ValueError):
#     AMTC = csdl.transforms.AMTC()
#     rv_dict = {
#         x_0: csdl.Variable(value=np.array([[[1.0], [2.0]],[[3.0], [4.0]]])),
#         # x_1: csdl.Variable(value=np.array([[[1.0], [2.0]],[[3.0], [4.0]]])+0.1),
#         # y: csdl.Variable(value=np.array([[[5.0], [6.0]],[[7.0], [8.0]],[[7.0], [8.0]]])),
#     }

#     tensor_grid_evalations = AMTC.transform(
#         rvs = rv_dict,
#         outputs = [z, y_out],
#     )
#     inputs = list(rv_dict.values())
#     outputs = list(tensor_grid_evalations.values())

#     # Try JAX sim
#     run_jax_sim(inputs, outputs, recorder)

def test_amtc_transform_tensor_2rv_2():
    recorder = csdl.Recorder(inline = False, debug=True)
    recorder.start()
    # a = csdl.Variable(value=np.array([3.0]), name = 'a')
    x_0 = csdl.Variable(value=np.array([[1.0], [1.0]]), name = 'x_0')
    x_1 = csdl.Variable(value=np.array([[1.0], [1.0]]), name = 'x_1')
    x = (x_1*2.0).add_name('x_1_scaled')
    x = (x_0+x).add_name('x_sum')
    # y = csdl.Variable(value=np.array([[1.0], [1.0]]), name = 'y')

    x_out = f10(x*x_1)
    # y_out = f01(y*x_1)
    # z = f11(x_out, y_out)
    # z_fake = (a*z)**3.0+2.0

    # recorder.visualize_graph(visualize_style='hierarchical')

    # Setup:
    # with pytest.raises(ValueError):
    AMTC = csdl.transforms.AMTC()
    rv_dict = {
        x_0: csdl.Variable(value=np.array([[[1.0], [2.0]],[[3.0], [4.0]]])),
        # x_1: csdl.Variable(value=np.array([[[1.0], [2.0]],[[3.0], [4.0]]])+0.1),
        # y: csdl.Variable(value=np.array([[[5.0], [6.0]],[[7.0], [8.0]],[[7.0], [8.0]]])),
    }

    tensor_grid_evalations = AMTC.transform(
        rvs = rv_dict,
        outputs = [x_out],
        # debug = True,
    )
    inputs = list(rv_dict.values())
    outputs = list(tensor_grid_evalations.values())

    # Try JAX sim
    run_jax_sim(inputs, outputs, recorder)


def run_jax_sim(inputs:list, outputs:list, recorder=None):
    jax_sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs=inputs,
        additional_outputs=outputs,
    )
    jax_sim.run()
    jax_sim.compute_totals()

    for out in outputs:
        print(f"{out.name} ({out.shape}): {np.mean(jax_sim[out])}")

if __name__ == '__main__':
    test_amtc_simple()
    test_amtc_simple2()
    test_amtc_transform()
    test_amtc_transform_tensor_1rv()
    test_amtc_transform_tensor_2rv()
    test_amtc_transform_tensor_2rv_2()