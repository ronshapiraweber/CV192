# /usr/bin/env python
"""
Created on Sat Mar 18 16:49:01 2017

Author: Oren Freifeld
Email: orenfr@cs.bgu.ac.il
"""
import os
import plt # this is the local plt pkg.
from __Timer__ import Timer

def inside_spyder():
    return any(['SPYDER' in name for name in os.environ])
