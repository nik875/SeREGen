import contextlib


def suppress_output(func: callable):
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            with contextlib.redirect_stdout(devnull):
                return func()
