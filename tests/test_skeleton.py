#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pyplot_assistant.skeleton import fib

__author__ = "clax.n"
__copyright__ = "clax.n"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
