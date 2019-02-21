# /usr/bin/env python
"""
Created on Sat Mar 18 19:19:33 2017

Author: Oren Freifeld
Email: orenfr@cs.bgu.ac.il
"""
import timeit
 
        
class Timer(object):
    def __init__(self):
        self.tic_was_called = False
        self.toc_was_called = False
    def tic(self):
        self.tic_was_called = True
        self.tic=timeit.default_timer()
    def toc(self):
        self.toc=timeit.default_timer()
        self.toc_was_called = True
        
        self.secs = (self.toc-self.tic)
    def __repr__(self):
        if self.tic_was_called and self.toc_was_called:
            return 'Timer: secs = {0}'.format(self.secs)
        else:
            return 'Unused Timer'
        
if __name__ == "__main__":
    import numpy as np
    timer = Timer()
   
    timer.tic()
    np.random.random((100000000))
    timer.toc()
    print timer
    