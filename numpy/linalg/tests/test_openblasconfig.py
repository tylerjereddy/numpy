from __future__ import division, absolute_import, print_function

import pytest
import platform
from distutils.version import LooseVersion

import numpy as np
from numpy.distutils import system_info

# TODO: more graceful check for openblas
@pytest.mark.skipif(len(system_info.get_info('openblas')) < 2,
                     reason="Requires openblas")
def test_openblas_info_return_type():
    # openblas_info should return a bytes
    # object containing information about
    # the openblas that was linked to NumPy;
    # this will include a version number
    # after 0.3.4
    from numpy.linalg import openblas_config
    actual = openblas_config._openblas_info()
    assert isinstance(actual, bytes)

# TODO: more specific tests, but have to be cautious
# of the need for openblas during linking & specific
# versions for the formatting of the info string

@pytest.mark.skipif(len(system_info.get_info('openblas')) < 2,
                     reason="Requires openblas")
def test_openblas_version_string():
    # DEBUG only -- probing the config string
    # in CI, where openblas >= 0.3.4 is available
    # on Windows
    from numpy.linalg import openblas_config
    actual = openblas_config._openblas_info()
    # currently the string looks like this on Windows Azure:
    # b'OpenBLAS 0.3.5.dev DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=24'

    # on Linux Azure with older openblas it looks like this:
    # b'NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Nehalem'
    #assert b"0.3.3" in actual

@pytest.mark.skipif(len(system_info.get_info('openblas')) < 2,
                     reason="Requires openblas")
def test_openblas_get_info():
    openblas_dict = system_info.get_info('openblas')
    #if 'version' in openblas_dict.keys():
    # true for openblas >= 0.3.4
    version = openblas_dict['version']
    # use LooseVersion for alphanumeric handling;
    # at time of writing we have 0.3.5.dev
    assert LooseVersion(version) >= LooseVersion("0.3.4")
