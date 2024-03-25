if __name__ == '__main__':
    import csdl_alpha as csdl

    recorder = csdl.build_new_recorder()
    recorder.start()
    csdl.print_all_recorders()
    recorder2 = csdl.build_new_recorder()
    recorder2.start()
    csdl.print_all_recorders()
    recorder2.stop()
    recorder.stop()
    csdl.print_all_recorders()

    recorder.start()
    csdl.enter_namespace('a')
    x = csdl.Variable()

    csdl.enter_namespace('b')
    y = x * 2
    csdl.exit_namespace()

    csdl.enter_namespace('c')
    z = y*x
    csdl.exit_namespace()

    csdl.exit_namespace()

    recorder.stop()