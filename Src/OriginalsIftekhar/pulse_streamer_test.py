import numpy as np
import time
import os
import matplotlib.pyplot as plt

import TimeTagger

ip_hostname = '169.254.8.2'
pulser = PulseStreamer(ip_hostname)