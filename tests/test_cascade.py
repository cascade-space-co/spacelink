import math
import pytest
import numpy as np
import yaml
import os
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

from spacelink.cascade import Sink, Source, Stage


def test_basic_cascade():
    transmitter = Source()
    receiver = Sink()
    stage = Stage()

    stage.input = transmitter.output
    receiver.input = stage.output

    result = receiver.output()
    
