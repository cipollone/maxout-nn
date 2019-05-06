#!/usr/bin/env python3

'''\
Python script that removes all logs directories from this directory.
Logs are directories that are named with integers.
'''

import os
import shutil

logdir = os.path.dirname(__file__)
if logdir == '': logdir = '.'
logs = [os.path.join(logdir, l) for l in os.listdir(logdir) if l.isdigit()]
logs = [l for l in logs if os.path.isdir(l)]

for l in logs:
  shutil.rmtree(l)
