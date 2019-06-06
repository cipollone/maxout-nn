#!/usr/bin/bash
# Forwards all arguments to maxout.py

# Suppress TF messages. 0,1,2,3 - all,info,warn,error
export TF_CPP_MIN_LOG_LEVEL=1

# Suppress warnings generated by TF itself
python maxout.py "$@" |& \
	sed '/WARNING:tensorflow:From [^:]*site-packages\/tensorflow/,+2d'