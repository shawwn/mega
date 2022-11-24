import math
import npnd
import numpy as np
import torchvis as tv
from pprint import pprint as pp

m1 = npnd.values((2,2)) + 1
m2 = npnd.values((2,2)) + 1 + 4

def pr(x, *args):
  print(x, *args)
  tv.see(x, *args)
  return x

pr(m1@m2)

s1 = npnd.unstack(m1, axis=0)
s2 = npnd.unstack(m1, axis=1)

pr(np.array(
     [[(s1[0] * s2[0]).sum(), (s1[0] * s2[1]).sum()],
      [(s1[1] * s2[0]).sum(), (s1[1] * s2[1]).sum()]]))
 

def gridmul(a, b):
  u = npnd.unstack(a, axis=0)
  v = npnd.unstack(b, axis=1)
  out = []
  for i in range(len(u)):
    row = []
    for j in range(len(v)):
      row.append((u[i] * v[j]).sum())
    out.append(row)
  return np.array(out)

pr(gridmul(m1, m2))

class MegaMat:
  def __init__(self, val):
    self.val = np.array(val)

  @classmethod
  def values(cls, shape):
    return cls(npnd.values(shape))

  @property
  def shape(self):
    return self.val.shape

  def __array__(self):
    return self.val

  def __matmul__(self, other):
    #return self.val @ other
    return gridmul(self.val, other)

  def __getitem__(self, coord):
    return self.val[coord]

  def __repr__(self):
    return repr(self.val)


def matmul(a, b):
  a_shape = np.shape(a)
  b_shape = np.shape(b)
  out = []
  for i in range(a_shape[0]):
    row = []
    for j in range(b_shape[1]):
      total = 0
      for k in range(a_shape[0]):
        total += a[i][k] * b[k][j]
      row.append(total)
    out.append(row)
  return np.array(out)


def blockmul(a, b):
  a_shape = np.shape(a)
  b_shape = np.shape(b)
  assert a_shape[0] % 2 == 0
  assert b_shape[1] % 2 == 0

def tex(mat, u, v):
  shape = np.shape(mat)
  i = int(u * shape[0] + 0.0)
  j = int(v * shape[1] + 0.0)
  return i, j

def slab(n):
  return np.linspace(0, n, n, endpoint=False) / n

def slabs(n, m):
  a = slab(n)
  b = slab(m)
  a = a.reshape((1, n))
  b = b.reshape((m, 1))
  a, b = np.broadcast_arrays(a, b)
  return a, b

def sample(mat, n, m):
  return mat[
      (slabs(n, m)[1] * np.shape(mat)[1]).astype(int),
      (slabs(n, m)[0] * np.shape(mat)[0]).astype(int)]


pr(sample(m1, 2, 2) @ sample(m2, 2, 2) / 1)
pr(sample(m1, 4, 4) @ sample(m2, 4, 4) / 8)
pr(sample(m1, 8, 8) @ sample(m2, 8, 8) / 64)

pr((sample(m1, 2, 2) / 1) @ (sample(m2, 2, 2) / 1) / 1)
pr((sample(m1, 4, 4) / 2) @ (sample(m2, 4, 4) / 2) / 2)
pr((sample(m1, 8, 8) / 4) @ (sample(m2, 8, 8) / 4) / 4)

pr((sample(m1, 2, 2) / 1**1.5) @ (sample(m2, 2, 2) / 1**1.5))
pr((sample(m1, 4, 4) / 2**1.5) @ (sample(m2, 4, 4) / 2**1.5))
pr((sample(m1, 8, 8) / 4**1.5) @ (sample(m2, 8, 8) / 4**1.5))

M1, M2 = (sample(m1, 8, 8) / 4**1.5), (sample(m2, 8, 8) / 4**1.5)


def make_morton_table(n=2048):
  # // generate the morton table that's used to address individual
  # // ubertexture tiles.
  # unsigned int mortonTable[ 2048 ];
  mortonTable = [0] * n
  # for ( unsigned int i = 0; i < 2048; ++i )
  # {
  for i in range(n):
    # mortonTable[ i ] = 0;
    mortonTable[i] = 0
    # unsigned int mask = 1;
    mask = 1
    # for ( unsigned int b = 0; b < 11; ++b, mask += mask )
    #   mortonTable[ i ] |= ( i & mask ) << b;
    b = 0
    while b < math.log2(n):
      mortonTable[ i ] |= ( i & mask ) << b;
      b += 1
      mask += mask
  # }
  return mortonTable
  
def bits(x):
  assert 0 <= x < 2**32
  return f'{x:032b}'

def interleave_bits(x, y):
  bx = bits(x)
  by = bits(y)
  out = ''
  for i in range(len(bx)):
    out += bx[i]
    out += by[i]
  return out

def coord2morton(x, y):
  return int(interleave_bits(x, y), 2)

def morton2coord(v):
  b = bits(v)
  x = int(b[0::2], 2)
  y = int(b[1::2], 2)
  return x, y
  

def tt(x, axis=0, ndim=2):
  x = np.asarray(x)
  if len(np.shape(x)) < ndim:
    x = x.reshape([(1 if i == axis else -1) for i in range(ndim)])
  return x

def split2x2(m):
  m = tt(m)
  w, h = np.shape(m)
  lh, rh = np.split(m, 2)
  ul, ll = np.split(lh, 2, axis=1)
  ur, lr = np.split(rh, 2, axis=1)
  return [[ul, ll],
          [ur, lr]]


S1, S2 = split2x2(M1), split2x2(M2)

def splitmul(a, b):
  a = tt(a)
  b = tt(b)
  if len(np.shape(a)) <= 2: a = split2x2(a)
  if len(np.shape(b)) <= 2: b = split2x2(b)
  a_shape = np.shape(a)
  b_shape = np.shape(b)
  out = []
  for i in range(a_shape[0]):
    row = []
    for j in range(b_shape[1]):
      total = 0
      for k in range(a_shape[0]):
        total += a[i][k] @ b[k][j]
      row.append(total)
    out.append(row)
  return out
  
