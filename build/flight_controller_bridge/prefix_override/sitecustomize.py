import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/poorcsky/csky_ws/install/flight_controller_bridge'
