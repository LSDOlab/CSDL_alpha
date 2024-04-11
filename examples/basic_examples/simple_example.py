'''
Simple example:
'''
if __name__ == '__main__':
    import csdl_alpha as csdl

    recorder = csdl.Recorder()
    recorder.start()
    csdl.print_all_recorders()
    recorder2 = csdl.Recorder()
    recorder2.start()
    csdl.print_all_recorders()
    recorder2.stop()
    recorder.stop()
    csdl.print_all_recorders()

    recorder.start()
    csdl.enter_namespace('a')
    x = csdl.Variable((1,))

    csdl.enter_namespace('b')
    y = x * 2
    csdl.exit_namespace()

    csdl.enter_namespace('c')
    z = y*x
    csdl.exit_namespace()

    csdl.exit_namespace()

    recorder.stop()