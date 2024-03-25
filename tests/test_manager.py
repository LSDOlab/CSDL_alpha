import pytest

def test_one_instance_error():
    """
    throws an error if multiple manager instances are created
    """

    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()

    from csdl_alpha.manager import RecManager

    with pytest.raises(Exception) as e_info:
        manager = RecManager()

def test_wrong_order_error():
    """
    throws an error is an outer recorder tries to stop before an inner recorder
    """

    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()

    import csdl_alpha as csdl
    from csdl_alpha.api import manager

    recorder = csdl.build_new_recorder()
    recorder.start()
    recorder2 = csdl.build_new_recorder()
    recorder2.start()

    with pytest.raises(Exception) as e_info:
        recorder.stop()

def test_manager_information():
    """
    check to make sure manager works correctly
    """

    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()
    
    import csdl_alpha as csdl

    recorder = csdl.build_new_recorder()
    from csdl_alpha.api import manager
    assert manager.active_recorder is None  # No recorders active now
    
    # test one recorder
    assert len(manager.constructed_recorders) == 1
    assert len(manager.recorder_stack) == 0 # no active recorder yet until start
    assert manager.constructed_recorders[0] == recorder

    # test one recorder start
    recorder.start()
    assert len(manager.constructed_recorders) == 1
    assert len(manager.recorder_stack) == 1 # start activates recorder
    assert manager.constructed_recorders[0] == recorder # only one constructed recorder
    assert manager.recorder_stack[-1] == recorder # active recorder is the only one
    assert csdl.get_current_recorder() == recorder # get_current_recorder returns active recorder

    # test additional recorder
    recorder2 = csdl.build_new_recorder()
    assert len(manager.constructed_recorders) == 2 # two recorders now
    assert len(manager.recorder_stack) == 1 # still only 1 recorder
    assert manager.constructed_recorders[1] == recorder2
    assert manager.recorder_stack[-1] == recorder # active recorder is the only one
    assert csdl.get_current_recorder() == recorder # get_current_recorder returns active recorder

    # start recorder 2
    recorder2.start()
    assert len(manager.constructed_recorders) == 2 # two recorders still
    assert len(manager.recorder_stack) == 2 # 2 recorders active now
    assert manager.recorder_stack[-1] == recorder2 # active recorder is recorder 2
    assert csdl.get_current_recorder() == recorder2 # get_current_recorder returns active recorder

    # destroy all recorders
    recorder2.stop()
    recorder.stop()
    assert len(manager.constructed_recorders) == 2 # two recorders still
    assert len(manager.recorder_stack) ==0  # No recorders active now
    assert manager.active_recorder is None  # No recorders active now