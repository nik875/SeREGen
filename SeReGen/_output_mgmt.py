"""
Output management functions. Private mmodule, should not be accessed outside of the package.
"""
import os
import contextlib


def suppress_output(func: callable):
    """
    Call the function with no arguments and suppress its output.
    """
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            with contextlib.redirect_stdout(devnull):
                return func()

