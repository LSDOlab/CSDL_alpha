import pytest

def test_loop_slice():
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()
    from csdl_alpha.src.operations.set_get.loop_slice import _loop_slice, VarSlice
    from csdl_alpha.utils.inputs import variablize

    import csdl_alpha as csdl
    csdl.Recorder().start()

    from csdl_alpha.src.graph.variable import Variable

    int_var = Variable(name='iter1', value=1.0)
    int_var2 = Variable(name='iter2', value=2.0)


    # assert _loop_slice[int_var].tuple == (int_var,)
    # assert _loop_slice[[0,int_var,2]].tuple == ([0,int_var,2],)
    # assert _loop_slice[int_var:1].tuple == (slice(int_var,1),)

    # assert _loop_slice[0:int_var,0:1].tuple == (slice(0,int_var), slice(0,1))
    # assert _loop_slice[[0,int_var],0:1].tuple == ([0,int_var], slice(0,1))
    # assert _loop_slice[0:1,[0,int_var]].tuple == (slice(0,1), [0,int_var])
    # assert _loop_slice[[0,int_var],[0,1]].tuple == ([0,int_var], [0,1])

    var_slice = _loop_slice[0:1]
    assert tuple(var_slice.slices) == (slice(0,1),)
    assert var_slice.evaluate() == (slice(0,1),)
    assert var_slice.vars == ()

    var_slice = _loop_slice[int_var]
    assert tuple(var_slice.slices) == (int_var,)
    assert var_slice.evaluate(2) == (2,)
    assert var_slice.evaluate(3) == (3,)
    assert var_slice.vars == (int_var,)

    var_slice = _loop_slice[int_var, int_var2]
    assert tuple(var_slice.slices) == (int_var,int_var2)
    assert var_slice.evaluate(2,3) == (2,3)
    assert var_slice.evaluate(2,4) == (2,4)
    assert var_slice.vars == (int_var,int_var2)

    var_slice = _loop_slice[[int_var], int_var2]
    assert tuple(var_slice.slices) == (int_var,int_var2)
    assert var_slice.evaluate(2,3) == (2,3)
    assert var_slice.evaluate(2,4) == (2,4)
    assert var_slice.vars == (int_var,int_var2)

    var_slice = _loop_slice[int_var, 0, int_var2]
    assert tuple(var_slice.slices) == (int_var, 0,int_var2)
    assert var_slice.evaluate(2,3) == (2,0,3)
    assert var_slice.evaluate(1,6) == (1,0,6)
    assert var_slice.vars == (int_var,int_var2)

    temp1 = int_var-1
    temp2 = int_var+1
    var_slice = _loop_slice[[int_var2, 0 ,temp1],[int_var2] , 0:10:2, int_var2]
    assert tuple(var_slice.slices) == ([int_var2, 0 ,temp1],int_var2, slice(0,10,2),int_var2)
    with pytest.raises(IndexError):
        var_slice.evaluate(2,3,4) # only two variables
    assert var_slice.evaluate(2,3) == ([2, 0 ,3],2, slice(0,10,2),2)
    assert var_slice.evaluate(4,1) == ([4, 0 ,1],4, slice(0,10,2),4)
    assert var_slice.vars == (int_var2, temp1)

    # ====TODO: maybe add feature for later. NO SLICING W/ CSDL VARIABLES FOR NOW!
    with pytest.raises(TypeError):
        var_slice = _loop_slice[0:int_var]

    # ==== maybe add feature for later. NO SLICING W/ CSDL VARIABLES FOR NOW!
    with pytest.raises(TypeError):
        var_slice = _loop_slice[int_var2, int_var:temp2]

    # ==== maybe add feature for later. NO SLICING W/ CSDL VARIABLES FOR NOW!
    with pytest.raises(TypeError):
        _loop_slice[0,1,int_var2:int_var2:int_var2]

    # ==== maybe add feature for later. NO SLICING W/ CSDL VARIABLES FOR NOW!
    with pytest.raises(TypeError):
        _loop_slice[[int_var2, 0 ,int_var], temp1:temp2:2]
        
    return

    var_slice = _loop_slice[0:int_var]
    assert isinstance(var_slice, VarSlice)

    var_slice = _loop_slice[0:int_var]
    assert tuple(var_slice.slices) == (slice(0,int_var),)
    assert var_slice.evaluate(1) == (slice(0,1),)
    assert var_slice.evaluate(2) == (slice(0,2),)
    assert var_slice.vars == (int_var,)


    var_slice = _loop_slice[temp1:temp2]
    assert tuple(var_slice.slices) == (slice(temp1,temp2),)
    assert var_slice.evaluate(2,3) == (slice(2,3),)
    assert var_slice.evaluate(1,6) == (slice(1,6),)
    assert var_slice.vars == (temp1,temp2)

    var_slice = _loop_slice[int_var2, int_var:temp2]
    assert tuple(var_slice.slices) == (int_var2, slice(int_var,temp2),)
    assert var_slice.evaluate(1,2,3) == (1, slice(2,3),)
    assert var_slice.evaluate(5,1,6) == (5, slice(1,6),)
    assert var_slice.vars == (int_var2, int_var, temp2)

    var_slice = _loop_slice[[int_var2, 0 ,int_var], :temp2]
    assert tuple(var_slice.slices) == ([int_var2, 0 ,int_var], slice(None,temp2),)
    assert var_slice.evaluate(2,3,4) == ([2, 0 ,3], slice(None,4),)
    assert var_slice.evaluate(2,3,5) == ([2, 0 ,3], slice(None,5),)
    assert var_slice.vars == (int_var2, int_var, temp2)

    var_slice = _loop_slice[[int_var2, 0 ,int_var], temp1:temp2:2]
    assert tuple(var_slice.slices) == ([int_var2, 0 ,int_var], slice(temp1,temp2,2),)
    assert var_slice.evaluate(2,3,4,5) == ([2, 0 ,3], slice(4,5,2),)
    assert var_slice.evaluate(5,4,3,2) == ([5, 0 ,4], slice(3,2,2),)
    assert var_slice.vars == (int_var2, int_var, temp1, temp2)

    with pytest.raises(TypeError):
        _loop_slice[0,1,int_var2:int_var2:int_var2]

    var_slice = _loop_slice[[int_var2, 0 ,temp2],[temp1] , int_var2:temp2:2]
    assert tuple(var_slice.slices) == ([int_var2, 0 ,temp2],temp1, slice(int_var2,temp2,2),)
    with pytest.raises(IndexError):
        var_slice.evaluate(2,3,4,5) # only three variables
    assert var_slice.evaluate(2,3,2) == ([2, 0 ,3],2, slice(2,3,2),)
    assert var_slice.evaluate(4,1,2) == ([4, 0 ,1],2, slice(4,1,2),)
    assert var_slice.vars == (int_var2, temp2, temp1)

    # example
    # for i in csdl.range(10):
    #     y = x.get(_loop_slice[], slice_shape = shape)

def test_slices():
    """
    tests to make sure that slices are correct
    """
    from csdl_alpha.src.operations.set_get.loop_slice import _loop_slice as _slice

    assert _slice[0].evaluate() == (0,)
    assert _slice[[0,1,2]].evaluate() == ([0,1,2],)
    assert _slice[0:1].evaluate() == (slice(0,1),)

    assert _slice[0:1,0:1].evaluate() == (slice(0,1), slice(0,1))
    assert _slice[[0,1],0:1].evaluate() == ([0,1], slice(0,1))
    assert _slice[0:1,[0,1]].evaluate() == (slice(0,1), [0,1])
    assert _slice[[0,1],[0,1]].evaluate() == ([0,1], [0,1])

    with pytest.raises(TypeError):
        _slice[(0,1,(1,2,3))]

    assert  _slice[[1,2,3], [1,2,3]].evaluate() == ([1,2,3], [1,2,3])

    # List indices must be the same length
    with pytest.raises(IndexError):
        _slice[[1,2,3], [1,2]]

    with pytest.raises(IndexError):
        _slice[1, [1,2,3], [1,2]]

    with pytest.raises(IndexError):
        _slice[0:1, [1,2,3], [1,2]]

    with pytest.raises(IndexError):
        _slice[[1,2], [1,2], [2,3,4]]

    # weird cases :(
    assert _slice[1,[0,1],[0,1],1].evaluate() == (1, [0,1], [0,1], 1)
    with pytest.raises(IndexError):
        _slice[0:1,[0,1],[0,1],1, [1,2,3], [1,2,3], 0:1]

    assert _slice[1,[0,],[0,],1].evaluate() == (1, 0, 0, 1)

    import numpy as np
    with pytest.raises(TypeError):
        _slice[np.array([1, 2, 3])]
    # x = np.ones((10,9,8,7,6,5,4))
    # # test = x[[0,1],[0,1],0:1, [1,2,3], [1,2,3]]
    # test = x[[0,1],[0,1]]
    # print(test.shape)


    # x = np.ones((10,9,8,7,6,5,4))
    # x = np.ones((10,10,10,10,10,10,10))
    # test = x[0:3,[0,1],[0,1],0:5,[0,1]]
    # print(test.shape) # (2, 3, 5, 10, 10)
    # test = x[0:3,[0,1],[0,],[0,1],0:5]
    # print(test.shape) # (3, 2, 5, 10, 10)
    # test2 = x[0:3,[0,1],[0,0],[0,1],0:5]
    # print(test2-test) # (3, 2, 5, 10, 10)
    # test = x[0:3,[0,1],[0,1],[0,1],0:5]
    # print(test.shape) # (3, 2, 5, 10, 10)
    # test = x[0:3,[0,1],[0,1],0:5]
    # print(test.shape) # (3, 2, 5, 10, 10, 10)


def test_valid_indexing_integers():
    """
    tests to make sure that index checking is valid
    """
    from csdl_alpha.src.operations.set_get.utils import check_and_process_out_of_bound_slice
    from csdl_alpha.src.operations.set_get.slice import _slice

    three_d_shape = (10,9,8)
    one_d_shape = (1,)

    # === test too many indices ===
    slices = _slice[1,2,3,4]
    with pytest.raises(IndexError):
        check_and_process_out_of_bound_slice(slices, three_d_shape)

    slices = _slice[0,0]
    with pytest.raises(IndexError):
        check_and_process_out_of_bound_slice(slices, one_d_shape)

    # === test out of bounds integers ===
    slices = _slice[11,2,3]
    with pytest.raises(IndexError):
        check_and_process_out_of_bound_slice(slices, three_d_shape)

    slices = _slice[-1,2,3]
    with pytest.raises(IndexError):
        check_and_process_out_of_bound_slice(slices, three_d_shape)

def test_valid_indexing_slices():
    """
    tests to make sure that index checking is valid
    """
    from csdl_alpha.src.operations.set_get.utils import check_and_process_out_of_bound_slice
    from csdl_alpha.src.operations.set_get.slice import _slice
    import numpy as np

    three_d_shape = (10,9,8)
    one_d_shape = (1,)
    x_3d = np.arange(np.prod(three_d_shape)).reshape(three_d_shape)
    x_1d = np.arange(np.prod(one_d_shape)).reshape(one_d_shape)

    # === test too many indices ===
    slices = _slice[0:1,:2,3:,4:]
    with pytest.raises(IndexError):
        check_and_process_out_of_bound_slice(slices, three_d_shape)
    with pytest.raises(IndexError):
        x_3d[slices]

    slices = _slice[0:1,:2]
    with pytest.raises(IndexError):
        check_and_process_out_of_bound_slice(slices, one_d_shape)
    with pytest.raises(IndexError):
        x_1d[slices]

    # === test out of bounds slices ===
    slices = _slice[0:100,:2]
    check_and_process_out_of_bound_slice(slices, three_d_shape)
    x_3d[slices]

    slices = _slice[-100:100]
    check_and_process_out_of_bound_slice(slices, one_d_shape)
    x_1d[slices]

def test_valid_indexing_lists():
    # === test out of bounds slices ===
    from csdl_alpha.src.operations.set_get.utils import check_and_process_out_of_bound_slice
    from csdl_alpha.src.operations.set_get.slice import _slice
    import numpy as np

    three_d_shape = (10,9,8)
    one_d_shape = (1,)
    x_3d = np.arange(np.prod(three_d_shape)).reshape(three_d_shape)
    x_1d = np.arange(np.prod(one_d_shape)).reshape(one_d_shape)

    # === test too many indices ===
    slices = _slice[[0,1],[0,1],[0,1],[0,1]]
    with pytest.raises(IndexError):
        check_and_process_out_of_bound_slice(slices, three_d_shape)
    with pytest.raises(IndexError):
        x_3d[slices]

    slices = _slice[[0,],[0,]]
    with pytest.raises(IndexError):
        check_and_process_out_of_bound_slice(slices, one_d_shape)
    with pytest.raises(IndexError):
        x_1d[slices]

    # === test out of bounds lists ===
    slices = _slice[[0,1],[0,10],[0,1]]
    with pytest.raises(IndexError):
        check_and_process_out_of_bound_slice(slices, three_d_shape)
    with pytest.raises(IndexError):
        x_3d[slices]

    slices = _slice[[1,]]
    with pytest.raises(IndexError):
        check_and_process_out_of_bound_slice(slices, one_d_shape)
    with pytest.raises(IndexError):
        x_1d[slices]

    # === test out of bounds mix ===
    five_d_shape = (10,9,8,7,6)
    x_5d = np.arange(np.prod(five_d_shape)).reshape(five_d_shape)

    slices = _slice[[0,1],[0,5],[0,1]]
    check_and_process_out_of_bound_slice(slices, five_d_shape)
    x_5d[slices]

    slices = _slice[1,3,[0,1],[0,5],3:4]
    check_and_process_out_of_bound_slice(slices, five_d_shape)
    x_5d[slices]
    
if __name__ == '__main__':
    # test_slices_int_variable()
    test_loop_slice()
    test_slices()
    exit()
    test_valid_indexing_integers()
    test_valid_indexing_slices()
    test_valid_indexing_lists()
    exit()

    import numpy as np
    # x = np.zeros((10,9,8))
    # x[18,2,3]
    # print(x.shape)

    # x = np.arange(18).reshape((3,2,3))
    # print(x)
    # print(x[0:2,])
    # print(x[0:2])
    # print()
    # print(x.shape)
    # print(x[0:2,].shape)
    # print(x[0:2].shape)

    x = np.ones((10,))
    print(x[[1,2,3]])
    print(x[1,2,3])
    exit()


    from slice import _slice
    _slice[0,0,0]
    _slice[(0,0,0)]
    exit()


    # test_valid_indexing_integers()
    test_valid_indexing_slices()