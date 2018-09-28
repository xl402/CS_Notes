# Algorithms
## Merge Sort
```python
C = output[length=n]
A = 1st sorted array [n/2]
B = 2nd sorted array [n/2]
i = 1
j = 1

for k = 1 to n:
  if A(i)<B(j):
    C(k) =A(i)
    i++
  else[B(j)<A(i)]:
    C(k) =B(j)
    j++
```
## Inversion Counting
$\mathcal{O}(nlogn)$ method- divide and conquer:
```python
count(array A, length n):
  if n = 1 return 0
  else
    x = count(1st half of A, n/2)
    y = count(2nd half of A, n/2)
    z =countsplitinv(A, n)
    return x + y + z
```
