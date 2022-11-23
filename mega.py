import npnd
import numpy as np

m1 = npnd.values((2,2)) + 1
m2 = npnd.values((2,2)) + 1 + 4

print(m1@m2)

s1 = npnd.unstack(m1, axis=0)
s2 = npnd.unstack(m1, axis=1)

print(np.array(
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

print(gridmul(m1, m2))

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


print(sample(m1, 2, 2) @ sample(m2, 2, 2) / 1)
print(sample(m1, 4, 4) @ sample(m2, 4, 4) / 8)
print(sample(m1, 8, 8) @ sample(m2, 8, 8) / 64)

print((sample(m1, 2, 2) / 1) @ (sample(m2, 2, 2) / 1) / 1)
print((sample(m1, 4, 4) / 2) @ (sample(m2, 4, 4) / 2) / 2)
print((sample(m1, 8, 8) / 4) @ (sample(m2, 8, 8) / 4) / 4)


