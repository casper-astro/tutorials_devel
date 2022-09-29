import numpy as np

def toSigned(v, bits):
  mask = 1 << (bits-1)
  return -(v & mask) | (v & (mask-1))

def arm_snapshots(fpga):
  for ss in fpga.snapshots:
    ss.arm()

def read_snapshots(fpga):
  snapshot_data = {}
  for ss in fpga.snapshots:
    dat = ss.read(arm=False)['data']

    for (k,v) in dat.items():
      snapshot_data[k] = v

  return snapshot_data

def extract_raw_data(fpga):
  raw = read_snapshots(fpga)
  return raw

def deinterleave(words, samp_per_word, word_len=16, cx=False):
  x = []
  samp_per_word = samp_per_word if not cx else samp_per_word*2
  for w in words:
    offset = 0
    for s in range(0, samp_per_word):
      z = toSigned(0xffff & (w >> offset), word_len)
      offset+=word_len
      x.append(z)

  if cx:
    i = np.array(x[0::2])
    q = np.array(x[1::2])
    x = i + 1j*q

  return x

def form_data_mat(raw_data, el_order, Nsamps, samp_per_word, cx=True):

  Nel = len(el_order)
  X = np.zeros((Nel, Nsamps), dtype=np.complex128)
  for j, el in enumerate(el_order):
    r = raw_data[el]
    x = deinterleave(r, samp_per_word, cx=cx)
    X[j, :] = x
  return X

def get_rfsoc_data(fpga):
  arm_snapshots(fpga)
  fpga.write_int('snapshot_ctrl', 0)
  fpga.write_int('snapshot_ctrl', 1)
  raw = extract_raw_data(fpga)
  return raw


