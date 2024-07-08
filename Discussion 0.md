**1.** Let $X\in\mathbb{R}^{m\times n}$, we do not assume that $X$ is full rank.
(a) Give the definition of the rowspace, columnspace, and nullspace of $X$
- Rowspace: the span (linear combnitions) of the rows of $X$
- Columnspace: the span of the columns of $X$
- Nullspace: the set of vector $v$ such that $Xv=0$
(b) Check (write an informal proof for) the following facts

($b_{a}$) The rowspace of $X$ is the columnspace of $X^{T}$
**Proof:**
The rows of $X$ is the column of $X^{T}$, and vice versa

($b_{b}$) The nullspace of $X$ and the rowspace of $X$ are orthogonal complements
**Proof:**
the nullspace of $X$ is $\{v\in\mathbb{R}^{n}|Xv=\mathbf{0}\}$
that means $(r_{1},...,r_{n})v=\mathbf{0}$
$\therefore \forall r_{i}\in$rowspace of $X$, $r_{i}\cdot v=0$
which means the rowspace of $X$ and the nullspace of $X$ are orthogonal complements

($b_{c}$) The nullspace of $X^{T}X$ is the same as the nullspace of $X$
**Proof:**
If $v$ is the nullspace of $X$, then $X^{T}Xv=X^{T}\mathbf{0}=\mathbf{0}$. On the other hand, if $v'$ is the nullspace of $X^{T}X$, then $v^{T}X^{T}Xv=\mathbf{0}$
that means $(Xv)^{T}(Xv)=\mathbf{0}$
$\therefore Xv=\mathbf{0}$

($b_{d}$) The columnspace and rowspace of $X^{T}X$ are the same, and are equal to the rowspace of $X$
$X^{T}X$ is symmetric, then the row and column of $X^{T}X$ are the same, thus the rowspace and columnspace are the same. and because part (b) (c):
rowspace($X^{T}X$) = nullspace($(X^{T}X)^{\bot}$) = nullspace($X^{\bot}$) = rowspace($X$)


**2.** There are $n$ archers all shooting the same target (bulls-eye) of a radius 1. Let the score for a particular archer be defined to be the distance away from the center (the lower the score, the better, and 0 is the optimal score). Each ancher's score is independent of the others, and is diatributed uniformly between 0 and 1. What is the expected value of the worst (highest) score?
(a) Define a random variable $Z$ that corresponds with the worst (highest) score.
$Z=\max\{z_{1},...z_{n}\}$

(b) Derive the Cumulative Distribution Function (CDF) of $Z$
$F(z)=P(Z<z)=P(X_{1}<Z)P(X_{2}<Z)...P(X_{n}<Z)=\Pi_{i=1}^{n}P(X_{i}\le z)$
$F(z) = \begin{cases} 0 \quad z<0 \\ z^{n} \quad 0\le z \le 1 \\ 1 \quad z>1 \end{cases}$

(c) Derive the Probability Density Function (PDF) of $Z$
$f(z) = \frac{dF}{dz} = \begin{cases} nz^{n-1}\quad 0\le z\le1 \\ 0 \quad otherwise \end{cases}$

(d) Calculate the expect value of $Z$
$E[Z] = \int_{-\infty}^{\infty}zf(z)dz=\int_{0}^{1}znz^{n-1}=\frac{n}{n+1}$

**3.** Below, $\mathbf{x}\in\mathbb{R}^{d}$ means that $\mathbf{x}$ is a $d\times 1$ column vector with real-valued entries. Likewise, $\mathbf{A}\in\mathbb{R}^{d\times d}$ means that $\mathbf{A}$ is a $d\times d$ matrix with real-valued entries. 
Consider $\mathbf{x}, \mathbf{w}\in \mathbb{R}^{d}$ and $\mathbf{A}\in\mathbb{R}^{n\times n}$. Derive the following derivatives.
(a) $\frac{\partial w^{T}x}{\partial x}$ and$\triangledown_{x}(w^{T}x)$
$\frac{\partial w^{T}x}{\partial x}=w^{T}$
$\triangledown_{x}(w^{T}x)=w$

(b) $\frac{\partial w^{T}Ax}{\partial x}$ and $\triangledown_{x}(w^{T}Ax)$
$\frac{\partial w^{T}Ax}{\partial x}=w^{T}A$
$\triangledown_{x}(w^{T}Ax) = A^{T}w$

(c) $\frac{\partial w^{T}Ax}{\partial w}$ and $\triangledown_{w}(w^{T}Ax)$
$w^{T}Ax=Tr(w^{T}Ax)=Tr(Axw^{T})$
$\therefore\triangledown_{w}(w^{T}Ax)=Ax$
$\therefore \frac{\partial w^{T}Ax}{\partial w}=(\therefore\triangledown_{w}(w^{T}Ax))^{T}=x^{T}A^{T}$

(d) $\frac{\partial w^{T}Ax}{\partial A}$ and $\triangledown_{A}(w^{T}Ax)$
$w^{T}Ax=Tr(w^{T}Ax)=Tr(xw^{T}A)$
$\therefore\triangledown_{A}(w^{T}Ax)=wx^{T}$
$\therefore \frac{\partial w^{T}Ax}{\partial A}=(\therefore\triangledown_{A}(w^{T}Ax))^{T}=xw^{T}$

(e) $\frac{\partial x^{T}Ax}{\partial x}$ and $\triangledown_{x}(x^{T}Ax)$
$\triangledown_{x}(x^{T}Ax)=(A^{T}+A)x$
$\therefore \frac{\partial x^{T}Ax}{\partial x}=x^{T}(A+A^{T})$

(f) $\triangledown_{x}^{2}(x^{T}Ax)$
$\triangledown_{x}^{2}(x^{T}Ax) = \frac{d\triangledown_{x}(x^{T}Ax)}{dx}=A^{T}+A$
***

