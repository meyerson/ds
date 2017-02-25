#!/usr/bin/env python


'''
See http://docs.python.org/2/library/profile.html

Also, http://pythonhosted.org/line_profiler/  for line-level profiling
'''

import sys
import pstats

p = pstats.Stats(sys.argv[1])

# 'cumulative'
p.strip_dirs().sort_stats('time').print_stats()
