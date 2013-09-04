import corr
import time
import numpy as np

r = corr.katcp_wrapper.FpgaClient('chewie')
time.sleep(0.5)
r.progdev('tut7.bof')

r.write_int('a_is_input',1)
r.write_int('b_is_input',1)

print "configuring to send from GPIO_B to GPIO_A"
print ""
r.write_int('a_is_input',1)
r.write_int('b_is_input',0)

r.write_int('to_gpio_a',0)
r.write_int('to_gpio_b',0xffffffff)
time.sleep(0.01)

print 'A: 0 <------- B : 0xff'

from_a = r.read_int('from_gpio_a')
from_b = r.read_int('from_gpio_b')

print "Readback values: A: %s, B: %s " %(np.binary_repr(from_a,width=8),np.binary_repr(from_b,width=8))

print 'A: 0xff <------- B: 0x0'

r.write_int('to_gpio_a',0xffffffff)
r.write_int('to_gpio_b',0)
time.sleep(0.01)

from_a = r.read_int('from_gpio_a')
from_b = r.read_int('from_gpio_b')

print "Readback values: A: %s, B: %s " %(np.binary_repr(from_a,width=8),np.binary_repr(from_b,width=8))


print ""
print "configuring to send from GPIO_A to GPIO_B"
print ""

r.write_int('b_is_input',1)
r.write_int('a_is_input',0)

r.write_int('to_gpio_a',0)
r.write_int('to_gpio_b',0xff)
time.sleep(0.01)

print 'A: 0 -------> B: 0xff'

from_a = r.read_int('from_gpio_a')
from_b = r.read_int('from_gpio_b')

print "Readback values: A: %s, B: %s " %(np.binary_repr(from_a,width=8),np.binary_repr(from_b,width=8))

print 'A: 0xff -------> B: 0x0'

r.write_int('to_gpio_a',0xff)
r.write_int('to_gpio_b',0)
time.sleep(0.01)

from_a = r.read_int('from_gpio_a')
from_b = r.read_int('from_gpio_b')

print "Readback values: A: %s, B: %s " %(np.binary_repr(from_a,width=8),np.binary_repr(from_b,width=8))

