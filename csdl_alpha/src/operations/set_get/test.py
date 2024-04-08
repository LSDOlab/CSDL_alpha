import pytest

def test_slices():
    """
    tests to make sure that slices are correct
    """
    from csdl_alpha.utils.slice import _slice

    assert _slice[0] == (0,)
    assert _slice[[0,1,2]] == ([0,1,2],)
    assert _slice[0:1] == (slice(0,1),)

    assert _slice[0:1,0:1] == (slice(0,1), slice(0,1))
    assert _slice[[0,1],0:1] == ([0,1], slice(0,1))
    assert _slice[0:1,[0,1]] == (slice(0,1), [0,1])
    assert _slice[[0,1],[0,1]] == ([0,1], [0,1])

    with pytest.raises(TypeError):
        _slice[(0,1,(1,2,3))]

    assert  _slice[[1,2,3], [1,2,3]] == ([1,2,3], [1,2,3])

    # List indices must be the same length
    with pytest.raises(IndexError):
        _slice[[1,2,3], [1,2]]

    with pytest.raises(IndexError):
        _slice[1, [1,2,3], [1,2]]

    with pytest.raises(IndexError):
        _slice[0:1, [1,2,3], [1,2]]

    with pytest.raises(IndexError):
        _slice[[1,2], [1,2], [2]]

    # weird cases :(
    assert _slice[1,[0,1],[0,1],1] == (1, [0,1], [0,1], 1)
    with pytest.raises(IndexError):
        _slice[0:1,[0,1],[0,1],1, [1,2,3], [1,2,3], 0:1]

    assert _slice[1,[0,],[0,],1] == (1, 0, 0, 1)

    import numpy as np
    # x = np.ones((10,9,8,7,6,5,4))
    # # test = x[[0,1],[0,1],0:1, [1,2,3], [1,2,3]]
    # test = x[[0,1],[0,1]]
    # print(test.shape)


    x = np.ones((10,9,8,7,6,5,4))
    x = np.ones((10,10,10,10,10,10,10))
    test = x[0:3,[0,1],[0,1],0:5,[0,1]]
    print(test.shape) # (2, 3, 5, 10, 10)
    test = x[0:3,[0,1],[0,1],0:5]
    print(test.shape) # (3, 2, 5, 10, 10, 10)


def test_valid_indexing_integers():
    """
    tests to make sure that index checking is valid
    """
    from utils import check_and_process_out_of_bound_slice
    from csdl_alpha.utils.slice import _slice

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
    from utils import check_and_process_out_of_bound_slice
    from csdl_alpha.utils.slice import _slice
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
    from utils import check_and_process_out_of_bound_slice
    from csdl_alpha.utils.slice import _slice
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
    test_slices()
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


    from csdl_alpha.utils.slice import _slice
    _slice[0,0,0]
    _slice[(0,0,0)]
    exit()


    # test_valid_indexing_integers()
    test_valid_indexing_slices()