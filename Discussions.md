# Discussion 0
## Problem1
Let $X\in\mathbb{R}^{m\times n}$, we do not assume that $X$ is full rank.
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


## Problem2
There are $n$ archers all shooting the same target (bulls-eye) of a radius 1. Let the score for a particular archer be defined to be the distance away from the center (the lower the score, the better, and 0 is the optimal score). Each ancher's score is independent of the others, and is diatributed uniformly between 0 and 1. What is the expected value of the worst (highest) score?
(a) Define a random variable $Z$ that corresponds with the worst (highest) score.
$Z=\max\{z_{1},...z_{n}\}$

(b) Derive the Cumulative Distribution Function (CDF) of $Z$
$F(z)=P(Z<z)=P(X_{1}<Z)P(X_{2}<Z)...P(X_{n}<Z)=\Pi_{i=1}^{n}P(X_{i}\le z)$
$F(z) = \begin{cases} 0 \quad z<0 \\ z^{n} \quad 0\le z \le 1 \\ 1 \quad z>1 \end{cases}$

(c) Derive the Probability Density Function (PDF) of $Z$
$f(z) = \frac{dF}{dz} = \begin{cases} nz^{n-1}\quad 0\le z\le1 \\ 0 \quad otherwise \end{cases}$

(d) Calculate the expect value of $Z$
$E[Z] = \int_{-\infty}^{\infty}zf(z)dz=\int_{0}^{1}znz^{n-1}=\frac{n}{n+1}$


## Problem 3
Below, $\mathbf{x}\in\mathbb{R}^{d}$ means that $\mathbf{x}$ is a $d\times 1$ column vector with real-valued entries. Likewise, $\mathbf{A}\in\mathbb{R}^{d\times d}$ means that $\mathbf{A}$ is a $d\times d$ matrix with real-valued entries. 
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
# Discussion 1
## Problem1
(a) consider log-likelihood $l(\theta)=\log L(\theta)$ instead of $L(\theta)$ when performing MLE, Why does this yield the same solution $\hat{\theta}_{MLE}$? Why is it easier to solve the optimization problem for $l(\theta)$ in the iid case? Given the observations $y_{1}, y_{2}, ..., y_{n}$, write down both $L(\theta)$ and $l(\theta)$ for Gaussian $f_{\theta}(y)=\frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-\frac{(y-\mu)^{2}}{2\sigma^{2}}}$ with $\theta=(\mu,\sigma)$
$L(\theta)=\prod\limits_{i=1}^{n}\frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-\frac{(y_{i}-\mu)^{2}}{2\sigma^{2}}}$
$l(\theta)=\sum\limits_{i=1}^{n}\frac{1}{\sqrt{2\pi\sigma^{2}}}e^{-\frac{(y_{i}-\mu)^{2}}{2\sigma^{2}}}$
$\frac{\partial L(\theta)}{\partial \theta_{i}}=\frac{\partial L(\theta)}{\partial l(\theta)}\frac{\partial l(\theta)}{\partial \theta_{i}}=\frac{1}{L(\theta)}\frac{\partial l(\theta)}{\partial \theta_{i}}$
therefore when $\frac{\partial L(\theta)}{\partial \theta_{i}}=0$, $\frac{\partial l(\theta)}{\partial \theta_{i}}=0$
thus they yield the same solution $\hat{\theta}_{MLE}$
and since $\sum$ is easier than $\prod$ to process when calculating derivative, making log-likelihood a more preferred optimization function


(b) The Poisson distribution is $f_{\lambda}(y)=\frac{\lambda^{y}e^{-\lambda}}{y!}$, Let $Y_{1}, Y_{2},..., Y_{n}$ be a set of iid Rvs with Possion distribution with parameter $\lambda$. Find the joint distribution of $Y_{1}, Y_{2}, ..., Y_{n}$. Find the maximum likelihood estimator of $\lambda$ as a function of observations $y_{1}, y_{2}, ..., y_{n}$
the joint distribution of $Y_{1}, Y_{2}, ..., Y_{n}$ is:
$p([y_{1},y_{2},...,y_{n}]=[Y_{1},Y_{2},...,Y_{n}])=\prod\limits_{i=1}^{n}\frac{\lambda^{y_{i}}e^{-\lambda}}{y_{i}!}$
the log-likelihood then be: $\sum\limits_{i=1}^{n}{y_{i}\log\lambda-\lambda}-{\sum_{i=2}^{y_{i}}\log{i}}$
then $\frac{\partial p(y_{1},y_{2},...,y_{n})}{\partial \lambda}=\sum_\limits{i=1}^{n}{\frac{y_{i}}{\lambda}-1}=0$
therefore $\hat{\lambda}_{MLE}=\frac{\sum\limits_{i=1}^{n}y_{i}}{n}=\overline{Y}$


## Problem2
(a) Consider the RV $X$ and $Y$ in $R$ with the following conditions. Are they **uncorrelated**, **independent**?
(i) $X$ and $Y$ takes values $\{-1,0,1\}$
(ii) When $X$ is 0, $Y$ takes values 1 and -1 with equal probability. When $Y$ is 0, $X$ takes values 1 and $-1$ with equal probability
(iii) Either X is 0 with probability ($\frac{1}{2}$), or Y is 0 with probability ($\frac{1}{2}$)
That means $(x,y)$ takes 4 possible points with equal probability: $(0,-1),(0,1),(1,0),(-1,0)$
then the pdf of $X$ and $Y$'s joint distribution is $p_{XY}(x,y)=\begin{cases}\frac{1}{4}\quad(x,y)\text{ in }\{(0,-1),(0,1),(1,0),(-1,0)\}\\0\quad\text{otherwise}\end{cases}$
then $\text{cov}(X,Y)=\sum\limits_{x,y\text{ in }\{-1,0,1\}}xyp_{XY}(x,y)=0$, thus $X$ and $Y$ is uncorrelated
and we have $p(x)=p(y)=\begin{cases}\frac{1}{2}\quad 0\\\frac{1}{4}\quad \pm1\end{cases}$
e.g. $p_{X}(0)p_{Y}(0)=\frac{1}{4}\neq p_{XY}(0,0)=0$, thus $X$ and $Y$ are dependent


(b) For $X=[X_{1}, X_{2},...,X_{n}]^{T}\sim N(\mu,\Sigma)$, vertify that if $X_{i},x_{j}$ are independent (for all $i\neq j$), then $\Sigma$ must be diagonal, i.e. $X_{i}$ and $X_{j}$ are uncorrelated
if $X_{i}$ and $X_{j}$ are independent, then $p_{X_{i}X_{j}}(x_{i},x_{j})=p_{X_{i}}(x_{i})p_{X_{j}}(x_{j})$
then $\text{cov}(X_{i},X_{j})=E[(x_{i}-\mu_{i})(x_{j}-\mu_{j})]=E(x_{i}-\mu{i})E(x_{j}-\mu_{j})=0$
thus $\Sigma=\text{diag}([\sigma^{2}_{1},\sigma^{2}_{2},...,\sigma^{2}_{n}]^{T})$, $X_{i}$ and $X_{j}$ is uncorrelated for $i\neq j$


(c) Let $N=2$, $\mu=\left(\begin{matrix}0\\0\end{matrix}\right)$, $\Sigma=\left(\begin{matrix}\alpha&\beta\\\beta&\gamma\end{matrix}\right)$. Suppose $X=\left(\begin{matrix}X_{1}\\X_{2}\end{matrix}\right)\sim N(\mu,\Sigma)$. Show that $X_{1}$ and $X_{2}$ are independent if $\beta=0$.
$\text{cov}(X_{1},X_{2})=E(x_{1}x_{2})=\beta=0$
and we have $E(x_{1})=0, E(x_{2})=0$
that is $E(x_{1}x_{2})=E(x_{1})E(x_{2})$, i.e. $X_{1}$ and $X_{2}$ are independent


(d) Consider a data point x drawn from an $N$-dimensional zero mean Multivariate Gaussian distribution $N(0,\Sigma)$, as shown above. assume that $\Sigma^{-1}$ exists. Prove that there exists a matrix $A\in \mathbb{R}^{N\times N}$ such that $x^{T}\Sigma^{-1} x=||Ax||^{2}$
since $\Sigma$ is diagnoal, then there exists $Q\in\mathbb{R}^{N\times N}$, where $Q^{T}=Q^{-1}$, $QQ^{T}=I$, that $\Sigma=Q\Lambda Q^{T}$, where $\Lambda$ contains all the eigenvalues of $\Sigma$.
since $\Sigma$ is PSD, then all the eigenvalues are nonnegative
thus $\Sigma^{-1}=Q\Lambda^{-1}Q^{T}$, let $A^{T}=Q\Lambda^{-\frac{1}{2}}$, then $x^{T}\Sigma^{-1}x=x^{T}A^{T}Ax=(Ax)^{2}=||Ax||^{2}$


## Problem3
(a) In ordinary least-squares linear regression, we typically have $n>d$ so that there is no $w$ such $Xw=y$ (these are typically overdetermined systems — too many equations given the number of unknowns). Hence, we need to find an approximate solution to this problem. The residual vector will be $r = Xw-y$ and we wantto make it as small as possible. The most common case is to measure the residual error with the standard Euclidean $l^{2}$-norm. So the problem becomes:
$\min\limits_{w}||Xw-y||^{2}_{2}$, where $X\in\mathbb{R}^{n\times d}$, $w\in\mathbb{R}^{d}$, $y\in \mathbb{R}^{n}$
Derive using vector calculus an expression for an optimal estimate for $w$ for this problem assuming $X$ is full rank.
same as linear regression:
$\min\limits_{w}||Xw-y||^{2}_{2}=\min (Xw-y)^{T}(Xw-y)=\min w^{T}X^{T}w-y^{T}Xw-yw^{T}X^{T}+y^{T}y$
$\frac{\partial (w^{T}X^{T}w-y^{T}Xw-yw^{T}X^{T}+y^{T}y)}{\partial w}=2(w^{T}X^{T}-y^{T})X=0$
therefore $w=(XX^{T})^{-1}X^{T}y$


(b) How do we know that $X^{T}X$ is invertible?
because $\text{det}(X^{T}X)=det(X)^{2}>0$, thus $X^{T}X$ is invertible.


(c) What should we do if $X$ is not full rank?
reduce the features until the matrix left is revertable or $n<d$