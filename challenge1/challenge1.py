from math import exp


def Relu(x: float):
    if x > 0:
        return x
    else:
        return 0


x1 = 1
x2 = -3.1
x3 = -7.2
x4 = 2.1

X = [x1, x2, x3, x4]

w1 = 2.1
w2 = 1.2
w3 = 0.3
w4 = 1.3

W = [w1, w2, w3, w4]

b = -3


def lin_reg1(X, W, b):
    res = 0
    for idx, x in enumerate(X):
        res += x * W[idx]
    return res + b


res1 = Relu(lin_reg1(X, W, b))
print("res :", lin_reg1(X, W, b), "partie 1 activation :", res1)

w1 = 0.1
w2 = 1.2
w3 = 4.9
w4 = 3.1

b = -5

W = [w1, w2, w3, w4]

res2 = Relu(lin_reg1(X, W, b))
print("res :", lin_reg1(X, W, b), "partie 2 activation :", res2)

w1 = 0.4
w2 = 2.6
w3 = 2.5
w4 = 3.8

b = -8

W = [w1, w2, w3, w4]

res3 = Relu(lin_reg1(X, W, b))
print("res :", lin_reg1(X, W, b), "partie 3 activation :", res3)

X = [res1, res2, res3]

w1 = 1.1
w2 = -4.1
w3 = 0.7

W = [w1, w2, w3]

b = 5.1

def activation_next_layer(x):
    return 1/( 1 + exp(-x) )

res4 = activation_next_layer(lin_reg1(X, W, b))


print("res :", lin_reg1(X, W, b), "partie 4 activation :", res4)


