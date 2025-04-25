import numpy as np
import struct

# bram parameters - need to match our yellow block's values
block_size = 128     # <bram data_width>
bram_addr_width = 13 # <bram address_width>
blocks = 2**bram_addr_width  # number of bram blocks
bits_per_val = 16 # <rfdc input data size> 16 bits for rfsoc4x2
# We need our output data size to match the bram's
# capacity so we don't fail on writes
num_vals = int(block_size / bits_per_val * blocks)

# sine wave parameters
fs = 1966.08e6      # RFDC sampling frequency
fc = 393.216e6      # Carrier frequency
dt = 1/fs           # Time length between samples
tau = dt * num_vals # Time length of bram

# Print useful info
print(f"bram_size = 2**{bram_addr_width}")
print(f"fs = {fs / 1e6} MHz")
print(f"fc = {fc / 1e6} MHz")

# Setup our array
t = np.arange(0,tau,dt)

# Generate our sine wave
# frequency fc
# range 0, 1
x = 0.5*(1+np.cos(2*np.pi* fc *t))
# scale our function to use the whole DAC range
maxVal = 2**14-1
x *= maxVal
# set each value to a 16 bit integer, for DAC compatibility
x = np.round(np.short(x))
# Shift right, DAC is 14 bits
x <<= 2

# Save our array x as a set of bytes
buf = bytes()
for i in x:
  buf += struct.pack('>h',i)

# We're done!, we can now write buf to our
# bram. To make sure it exists, enter len(buf)
# in your ipython terminal
