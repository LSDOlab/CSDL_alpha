
def hard_reload():
    """
    resets the manager instance singleton.
    should be rarely used.
    """

    print('WARNING: resetting manager...')
    from importlib import reload
    
    from csdl_alpha.manager import RecManager
    RecManager.instantiated = False
    import csdl_alpha.api
    reload(csdl_alpha.api)