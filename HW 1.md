*I certify that all solutions are entirely in my own words and that I have not looked at another student’s solutions. I have given credit to all external sources I consulted*

# Problem 1
First, we defined the **derivative** and **gradient** of a function of a scalar/vector/matrix, which is as below
- for scalar, the gradient and the derivative are the same, which is $\frac{df}{dx}$
- for vector, the gradient  $[\triangledown_{\mathbf{X}}f(\mathbf{X})]_{i}=\frac{\partial f}{\partial X_{i}}(\mathbf{X})$, and the derivative is $[\triangledown_{\mathbf{X}}f(\mathbf{X})]^{T}$
- for matrix, the gradient: $[\triangledown_{\mathbf{X}}f(\mathbf{X})]_{ij}=\frac{\partial f}{\partial X_{ij}}(\mathbf{X})$, also, the derivative is $[\triangledown_{\mathbf{X}}f(\mathbf{X})]^{T}$
This means that the shape of gradient is the same as input, which allows us to compute the $i$th (or $ij$th) entry using the partial derivative of $f$ with the respect to the $i$th ($ij$th) entry of the input

Then, define the **Hessian** of a function $f : \mathbb{R}^{n} \rightarrow \mathbb{R}$ as the $n \times n$ matrix with elements
$$
\displaylines{
[\triangledown^{2}_{\mathbf{x}}f{(\mathbf{x})}]_{ij}=\frac{\partial^{2}f}{\partial x_{i}\partial x_{j}}(\mathbf{x})
}
$$
The Hessian equals the derivative of the gradient, which is $\triangledown^{2}_{\mathbf{x}}f(\mathbf{x})=\frac{df}{d\mathbf{x}}[\triangledown_{\mathbf{x}}f(\mathbf{x})]$
Same as the gradient provides the "best linear approximation" to a vector function $f(\mathbf{x})$ near a point $\mathbf{a} \in \mathbb{R}^{n}$, the Hessian can provide the "best quadratic approximation" as below
$$
\displaylines{
f(\mathbf{x}) = f(\mathbf{a}) + [\triangledown_{\mathbf{x}}f(\mathbf{a})]^{T}(\mathbf{x-a})+\frac{1}{2}(\mathbf{x-a})^{T}[\triangledown^{2}_{\mathbf{x}}f{(\mathbf{x})}](\mathbf{x-a})
}
$$

(a) $w \in \mathbb{R}^{n}$ , compute $\triangledown_{\mathbf{x}}f(\mathbf{a})$, where $f:\mathbb{R}^{n}\rightarrow\mathbb{R}, f(\mathbf{x}) = w^{T}\mathbf{x}$
**Solution:**
as defined above, $[\triangledown_{\mathbf{x}}f(\mathbf{a})]_{i} = \frac{\partial f}{\partial X_{i}}(\mathbf{X}) = w_{i}$
$\therefore$ $[\triangledown_{\mathbf{x}}f(\mathbf{a})] = w$


(b) $\mathbf{A}\in \mathbb{R}^{n\times n}$, compute $\triangledown_{\mathbf{x}}f(\mathbf{a})$, where $f:\mathbb{R}^{n}\rightarrow\mathbb{R}, f(\mathbf{x}) = \mathbf{x}^{T}A\mathbf{x}$
**Solution:**
also, $[\triangledown_{\mathbf{x}}f(\mathbf{a})]_{i} = \frac{\partial f}{\partial X_{i}}(\mathbf{X}) = 2(\sum_\limits{j=1}^{n}A_{ij}x_{j}) = 2A_{i}^{T}\mathbf{x}$
$\therefore$ $[\triangledown_{\mathbf{x}}f(\mathbf{a})] = 2A\mathbf{x}$


(c) compute the Hessian $\triangledown^{2}_{\mathbf{x}}f(\mathbf{x})$ of the function above
**Solution:**
the Hessian is the derivative of gradient, which is $\frac{df}{d\mathbf{x}}[2A\mathbf{x}] = [2A]^{T} = \frac{1}{2}A^{T}$


(d) $A\in\mathbb{R}^{m\times n}$, $y\in\mathbb{R}^{m}$, compute the gradient $\triangledown_{\mathbf{x}}f(\mathbf{x})$, where $f:\mathbb{R}^{n}\rightarrow\mathbb{R}, f(\mathbf{x}) = ||A\mathbf{x} - \mathbf{y}||^{2}$
**Solution:**
$f = ||\sum\limits_{i=1}^{n}\mathbf{a_{i}}x_{i}-y||$
$\therefore$  $[\triangledown_{\mathbf{x}}f(\mathbf{x})]_{i} = \frac{\partial f}{\partial X_{i}}(\mathbf{X}) = 2\mathbf{a_{1}}^{T}(A\mathbf{x}-\mathbf{y})$
$\therefore$ $\triangledown_{\mathbf{x}}f(\mathbf{x}) = 2A^{T}(A\mathbf{x}-\mathbf{y})$
*as shown in (e), this problem can also be solved using trace as below*
**Solution II:**
$f(\mathbf{x}) = ||A\mathbf{x} - \mathbf{y}||^{2}$
$=Tr((A\mathbf{x}-\mathbf{y})^{T}(A\mathbf{x}-\mathbf{y}))$
$=Tr((A\mathbf{x}-\mathbf{y})(A\mathbf{x}-\mathbf{y}))^{T}$
$=Tr(A\mathbf{x}\mathbf{x}^{T}A^{T}-y\mathbf{x}^{T}A^{T}-A\mathbf{x}\mathbf{y}^{T}+\mathbf{y}\mathbf{y}^{T})$
$\therefore \triangledown_{\mathbf{x}}f(\mathbf{x})=2A^{T}A-2A^{T}y = 2A^{T}(A\mathbf{x}-\mathbf{y})$


(e) $\mathbf{u}\in\mathbb{R}^{m}$, $\mathbf{v}\in\mathbb{R}^{n}$, compute the gradient $\triangledown_{\mathbf{x}}f(\mathbf{x})$, where $f:\mathbb{R}^{m\times n}\rightarrow\mathbb{R}, f(\mathbf{A}) = \mathbf{u}^{T}\mathbf{A}\mathbf{v}$
**Solution:**
$\because$ $f(\mathbf{A}) = \mathbf{u}^{T}\mathbf{A}\mathbf{v} = Tr(\mathbf{v}\mathbf{u}^{T}\mathbf{A})$
$\therefore$ $\triangledown_{\mathbf{x}}f(\mathbf{x}) = (\mathbf{v}\mathbf{u}^{T})^{T} = \mathbf{u}\mathbf{v}^{T}$


(f) $\mathbf{x}\in\mathbb{R}^{n}$, $\mathbf{y}\in\mathbb{R}^{m}$, compute the gradient $\triangledown_{\mathbf{x}}f(\mathbf{x})$, where $f:\mathbb{R}^{m\times n}\rightarrow\mathbb{R}, f(\mathbf{A}) = ||\mathbf{A}\mathbf{x}-\mathbf{y}||^{2}$
**Solution:**
same as (d), we use trace to solve this problem
$f(\mathbf{x}) = ||A\mathbf{x} - \mathbf{y}||^{2}$
$=Tr((A\mathbf{x}-\mathbf{y})^{T}(A\mathbf{x}-\mathbf{y}))$
$=Tr((A\mathbf{x}-\mathbf{y})(A\mathbf{x}-\mathbf{y}))^{T}$
$=Tr(A\mathbf{x}\mathbf{x}^{T}A^{T}-y\mathbf{x}^{T}A^{T}-A\mathbf{x}\mathbf{y}^{T}+\mathbf{y}\mathbf{y}^{T})$
$\therefore \triangledown_{\mathbf{x}}f(\mathbf{x}) = 2\mathbf{A}\mathbf{x}\mathbf{x}^{T} - 2\mathbf{y}\mathbf{x}^{T} = 2(A\mathbf{x}-\mathbf{y})\mathbf{x}^{T}$


(g)compute the gradient and Hessian of the LSE function: $f:\mathbb{R}^{n}\rightarrow\mathbb{R}$, $LSE(\mathbf{x})=\ln\left(\sum\limits^{n}_{i=1}e^{x_{i}}\right)$
**Solution:**
according to defination, we have:
$[\triangledown_{\mathbf{X}}f(\mathbf{X})]_{i}=\frac{\partial f}{\partial X_{i}}(\mathbf{X})=\frac{e^{x_{i}}}{\sum\limits^{n}_{i=1}e^{x_{i}}}$
$\therefore \triangledown_{\mathbf{X}}f(\mathbf{X}) = \frac{e^{\mathbf{x}}}{\sum\limits^{n}_{i=1}e^{x_{i}}}$
$\therefore [\triangledown_{\mathbf{X}}^{2}f(\mathbf{X})]_{ij} = \frac{\partial^{2} f}{\partial x_{i}\partial x_{j}}$ which equals:
$\begin{cases} -\frac{e^{x_{i}e_{x_{j}}}}{(\sum\limits^{n}_{i=1}e^{x_{i}})^{2}} \quad i\neq j \\ \frac{e^{x_{i}}}{\sum\limits^{n}_{i=1}e^{x_{i}}}-\frac{e^{x_{i}e_{x_{j}}}}{(\sum\limits^{n}_{i=1}e^{x_{i}})^{2}} i=j \end{cases}$
$\therefore \triangledown_{\mathbf{X}}^{2}f(\mathbf{X}) = diag(\frac{e^{\mathbf{x}}}{\sum\limits^{n}_{i=1}e^{x_{i}}})-\frac{e^{\mathbf{x}}(e^{\mathbf{x}})^{T}}{(\sum\limits^{n}_{i=1}e^{x_{i}})^{2}}$

# Problem 2

**1.** Let $\mathbf{A}$ be a symmetric matrix. Prove equilance between these different definitions of positive semidefiniteness (PSD)
(a) For all $\mathbf{x}\in\mathbb{R}^{n}$, $\mathbf{x}^{T}\mathbf{A}\mathbf{x}\ge0$
(b) All the eigenvalue of $\mathbf{A}$ are nonnegative
(c) There exists a matrix $U\in\mathbb{R}^{n\times n}$ such that $A=UU^{T}$

First, we prove (c) => (a)
$\because A=UU^{T}$
$\therefore \mathbf{x}^{T}\mathbf{A}\mathbf{x} = \mathbf{x}^{T}\mathbf{U}\mathbf{U}^{T}\mathbf{x}=(\mathbf{x^{T}U})(\mathbf{x^{T}U})^{T}$
$\therefore (\mathbf{x^{T}U})(\mathbf{x^{T}U})^{T} = \sum_{i=1}^{n}(\sum_{k=1}^{n}x_{k}U_{ik})^{2}\ge0$

Then, (a) => (b)
$\because \mathbf{A}$ is symmetric
$\therefore \mathbf{A}=\mathbf{P}\mathbf{\Lambda}\mathbf{P}^{T}$ where $\mathbf{\Lambda}$ is a diagonal matrix which contains all the eigenvalue $\lambda_{1},...\lambda{n}$, $\mathbf{P}$ is full rank
$\therefore \Lambda=\mathbf{P}^{-1}\mathbf{A}(\mathbf{P^{-1}})^{T}$
$\therefore \lambda_{i} = \mathbf{P}^{-1}_{i}\mathbf{A}(\mathbf{P}^{-1})^{T}_{i}\ge0$
which proved that all the eigenvalue of $\mathbf{A}$ are nonnegative

Finally, (b) => (c)
$\because \mathbf{A}$ is a symmetrical matrix
$\therefore \mathbf{A}=\mathbf{P}\mathbf{\Lambda}\mathbf{P}^{T}=\mathbf{P}diag(\sqrt{\lambda_{1}},...\sqrt{\lambda_{n}})diag(\sqrt{\lambda_{1}},...\sqrt{\lambda_{n}})\mathbf{P}^{T}$ 
and $\because  diag(\sqrt{\lambda_{1}},...\sqrt{\lambda_{n}})=diag(\sqrt{\lambda_{1}},...\sqrt{\lambda_{n}})^{T}$
then we have $\mathbf{A}=(\mathbf{P}diag(\sqrt{\lambda_{1}},...\sqrt{\lambda_{n}}))(\mathbf{P}diag(\sqrt{\lambda_{1}},...\sqrt{\lambda_{n}}))^{T}$
let $\mathbf{U}=\mathbf{P}diag(\sqrt{\lambda_{1}},...\sqrt{\lambda_{n}})$
then $\mathbf{A} = \mathbf{U}\mathbf{U}^{T}$
(c) proved


**2.** Prove the fellowing properties of PSD matrix
(a) If $\mathbf{A}$ and $\mathbf{B}$ are PSD, $2\mathbf{A}+3\mathbf{B}$ is PSD
For all $\mathbf{x} \in \mathbb{R}^{n} \mathbf{x}\mathbf{2A+3B}\mathbf{x}^{T}=2\mathbf{x}\mathbf{A}\mathbf{x}^{T}+3\mathbf{x}\mathbf{B}\mathbf{x}^{T}\ge0$
$\therefore$ $\mathbf{2A+3B}$ is PSD

(b) If $\mathbf{A}$ is PSD, then all the diagonal entries of $\mathbf{A}$ is nonnegative
$\because \forall \mathbf{x} \in \mathbb{R}^{n}$, $\mathbf{xAx}^{T}\ge 0$
then for $x_{i}$, construct standard basis vector $e_{i} = (0,...1,...0)^{T}$ where $1$ appears on the $i$-th entry
$\therefore e_{i}^{T}\mathbf{A}e_{i} = A_{ii} \ge0$

(c) If $\mathbf{A}$ is PSD, then the sum of all entries of $\mathbf{A}$ $\sum\limits_{i=1}^{n}\sum_{j=1}^{n}A_{ij}\ge0$
$\sum\limits_{i=1}^{n}\sum_{j=1}^{n}A_{ij} = (1,...1)\mathbf{A}(1,...1)^{T}\ge0$

(d) If $\mathbf{A}, \mathbf{B}$ are PSD, then $Tr(\mathbf{AB})\ge0$
$\because \mathbf{A} = \mathbf{PP}^{T}$, $\mathbf{B} = \mathbf{QQ}^{T}$
$\therefore \mathbf{AB} = \mathbf{PP}^{T}\mathbf{QQ}^{T}$
$\therefore Tr(\mathbf{AB})=Tr(\mathbf{PP}^{T}\mathbf{QQ}^{T})=Tr(\mathbf{Q}^{T}\mathbf{PP}^{T}\mathbf{Q})=Tr(\mathbf{(Q^{T}P)(Q^{T}P)^{T}})=||\mathbf{Q^{T}P}||^{2}\ge0$

(e) If $\mathbf{A}, \mathbf{B}$ are PSD, then $Tr(\mathbf{AB})=0$ if and only if $\mathbf{AB}=\mathbf{0}$
as metioned in (d), $Tr(\mathbf{AB})=||\mathbf{Q^{T}P}||^{2}=0$
then $\mathbf{Q^{T}P}=0, \mathbf{P^{T}Q}=0$
$\therefore \mathbf{AB} = \mathbf{PP^{T}QQ^{T}} = 0$


**3.** Let $\mathbf{A}\in\mathbb{R}^{n\times n}$ be a symmetric, PSD matrix. Write $||\mathbf{A}||_{F}$ as a funtion of the eigenvalues of $\mathbf{A}$
$||\mathbf{A}||_{F}=\sqrt{Tr(\mathbf{AA}^{T})}$


**4.** Let $\mathbf{A}\in\mathbb{R}^{n\times n}$ be a symmetric matrix. Prove that the largest eigenvalue of $\mathbf{A}$ is $\lambda_{max}(\mathbf{A})=\max\limits_{||x||_{2}=1}\mathbf{x^{T}Ax}$
$\mathbf{x^{T}Ax} = \sum\limits_{i=1}^{n}x_{i}^{2}\lambda_{i}\le \sum\limits_{i=1}^{n}x_{i}^{2}\lambda_{max}=\lambda_{max}$ only when $x=(0,...,1,...0)$, the $1$ appears at the $i_{max \lambda index}$-th entry


# Problem 3

**1.** Prove that the covariance matrix is always positive semidefinate
$\Sigma = \mathbb{E}[(Z-\mu)(Z-\mu)^{T}]$
for $\forall z\in\mathbb{R}^{n}$, we have $z^{T}\Sigma z=\mathbb{E}[z^{T}(Z-\mu)(Z-\mu)^{T}z]$
Let $(Z-\mu)^{T}z=Y$
then $z^{T}(Z-\mu)=Y^{T}$
then $\mathbb{E}[z^{T}(Z-\mu)(Z-\mu)^{T}z]=\mathbb{E}[Y^{T}Y]=\sum\limits_{i=1}^{n}\mathbb{E}[X_{i}]^{2}z_{i}^{2}\ge0$
That indicates $\forall z\in\mathbb{R}^{n}$, $z^{T}\Sigma z\ge0$, $\Sigma$ is positive semidefinite


**2.** The probability that an archer hits her target when it is windy is 0.4; when it is not windy, her probability of hitting the target is 0.7. On any shot, the probability of a gust of wind is 0.3. Find the probability that

(i) on a given shot there is a gust of wind and she hits her target.
$p=0.3*0.4=0.12$

(ii) she hits the target with her first shot.
$p=0.3*0.4+0.7*0.7=0.12+0.0.49=0.61$

(iii) she hits the target exactly once in two shots.
$p=2*0.61*(1-0.61)=0.4758$

(iv) there was no gust of wind on an occasion when she missed.
$p=\frac{0.3*(1-0.4)}{1-0.61}=0.462$


**3.** An archery target is made of $3$ concentric circles of radii $\frac{1}{\sqrt3}$, $1$ and $\sqrt3$ feet. Arrows striking within the inner circle are awarded 4 points, arrows within the middle ring are awarded 3 points, and arrows within the outer ring are awarded 2 points. Shots outside the target are awarded 0 points. 
Consider a random variable X, the distance of the strike from the center (in feet), and let the probability density function of X be
$$
\displaylines{
f(x)=\begin{cases} \frac{2}{\pi(1+x^{2})} \quad x>0 \\ 0\quad\quad\quad \text{otherwise} \end{cases}
}
$$
What is the expected value of the score of a single strike?
$\mathbb{E}(X)=\int_{0}^{\infty}xf(x)dx=\int_{0}^{\frac{1}{\sqrt3}}4f(x)dx + \int_{\frac{1}{\sqrt3}}^{1}3f(x)dx + \int_{1}^{\sqrt{3}}2f(x)dx$
$=\frac{3}{2}+\frac{2}{\pi}\arctan\sqrt3$

**4.** Let $X\sim\text{Pois}(\lambda)$, $Y\sim\text{Pois}(\mu)$. Given that $X$ and $Y$ are independent, derive an expression for $\mathbb{P}(X=k|X+Y=n)$ where $k=0,1,...,n$, then what well-known probability distribution is this?What are its parameters?
$\text{Poi}(X=k)=\frac{\lambda^{k}e^{-\lambda}}{k!}$
then $X+Y\sim\text{Poi}(\lambda+\mu)$
then $\mathbb{P}(X=k|X+Y=n)=\frac{P(X=k)P(Y=n-k)}{P(X+Y=k)}=\frac{\frac{\lambda^{k}e^{-\lambda}}{k!}\frac{\mu^{n-k}e^{-\mu}}{(n-k)!}}{\frac{(\lambda+\mu)^{n}e^{-(\lambda+\mu)}}{n!}}=\frac{n!}{k!(n-k)!}\left(\frac{\lambda}{\lambda+\mu}\right)^{k}\left(\frac{\mu}{\lambda+\mu}\right)^{n-k}$
it is a **Binomial Distribution** with parameters $n$ and $p=\frac{\lambda}{\lambda+\mu}$

# Problem 4

(a) Let $X\sim N(\mu,\Sigma)$. Show that $\mathbb{E}(X)=\mu$
**Proof:**
$E=\int xf(x)dx=\int_{\mathbb{R}^{d}}(t+\mu)f(t+\mu)=\mu+\int_{\mathbb{R}^{d}}t\frac{1}{\sqrt{(2\pi)^{d}|\Sigma|}}=\mu$


(b) Show that $\text{Cov(X)}=\Sigma$
First we have $\text{Cov}(X)=E((X-\mu)(X-\mu)^{T})$
$=\frac{1}{\sqrt{(2\pi)^{d}|\Sigma|}}\int (X-\mu)(X-\mu)^{T}\exp\left(-\frac{1}{2}(X-\mu)^{T}\Sigma(X-\mu)\right)dX$
$=\frac{1}{\sqrt{(2\pi)^{d}|\Sigma|}}\int YY^{T}\exp\left(-\frac{1}{2}Y^{T}\Sigma Y\right)dY$,  in which $Y=X+\mu$, $Y\sim N(\mathbf{0},\Sigma)$
$=\frac{1}{\sqrt{(2\pi)^{d}|\Sigma|}}\int \sqrt{\Sigma}ZZ^{T}\sqrt{\Sigma}\exp\left(-\frac{1}{2}Z^{T}Z\right)|\Sigma^{\frac{1}{2}}|dZ$,  in which $Z=\sqrt{\Sigma^{-1}}Y$, $Z\sim N(0,I)$
$=\sqrt{\Sigma}\left(\int\frac{1}{\sqrt{(2\pi)^{d}I}}ZZ^{T}\exp\left(-\frac{1}{2}Z^{T}IZ\right)\right)\sqrt{\Sigma}$
$=\sqrt{\Sigma}\mathbb{E}(ZZ^{T})\sqrt{\Sigma}$
$=\sqrt{\Sigma}I\sqrt{\Sigma}$
$=\Sigma$
i.e. $\text{Cov}(X)=\Sigma$


(c) Computing the moment generating function (MGF) of $X$: $M_{X}(\lambda)=\mathbb{E}[e^{\lambda^{T}X}]$, where $\lambda\in\mathbb{R}^{d}$
(*note that there are several interesting and usful properties of MGF, one being: if $M_{X}=M_{Y}$, then $X$ and $Y$ has the same distribution*)
$M_{x} = \mathbb{E}[e^{\lambda^{T}X}]=\int e^{\lambda^{T}X}f(X)dX$
$=\int \frac{1}{\sqrt{(2\pi)^{d}|\Sigma|}}\exp\left(\lambda^{T}x-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)\right)dx$
$=\frac{1}{\sqrt{(2\pi)^{d}|\Sigma|}}\int\exp\left( \lambda^{T}y-\frac{1}{2}y^{T}\Sigma^{-1}y\right)\exp(\lambda^{T}\mu)dy$,  $y=x-\mu$
$=\frac{e^{\lambda^{T}\mu}}{\sqrt{(2\pi)^{d}}}\int\exp\left(\lambda^{T}\Sigma^{\frac{1}{2}}z-\frac{1}{2}z^{T}z\right)dz$,  $z=\Sigma^{-\frac{1}{2}}y$
$=e^{\lambda^{T}\mu+\frac{1}{2}\lambda^{T}\Sigma\lambda}\int\frac{1}{\sqrt{(2\pi)^{d}}}\exp\left(-\frac{1}{2}\phi^{T}\phi\right)$,  $\phi=z-\Sigma^{\frac{1}{2}}\lambda$
$=e^{\lambda^{T}\mu+\frac{1}{2}\lambda^{T}\Sigma\lambda}$


(d) Using the fact that MGFs determing distributions, given $A\in\mathbb{R}^{k\times d}$, $b\in\mathbb{R}^{k}$ identify the distribution of $Ax+b$
$M_{Ax+b}=\int e^{\lambda^{T}(Ax+b)}f(x)dx$
$=e^{\lambda^{T}b}\int e^{\lambda^{T}Ax}f(x)dx$
$=e^{\lambda^{T}(A\mu+b)+\frac{1}{2}\lambda^{T}A\Sigma A^{T}\lambda}$
thus $Ax+b\sim N(A\mu+b,A\Sigma A^{T})$


(e) Show that there exists an affine transformation of $X$ that is distributed as the standard multivariate Gaussian, $N(0,I_{d})$
$\Sigma = P\Lambda P^{T}$
then for $Y=P^{-1}\Lambda^{-\frac{1}{2}}(X-\mu)$:
$Y\sim N(0,I_{d})$


# Problem 5

(a) Implement `special_reshape`, which takes an ndarray with an arbitrary number of dimensions and reduces it to 2 dimensions, so that the first $n − 1$ dimensions of the input get combined into the first output dimension, and the last dimension of the input gets preserved in the output. For example, an input ndarray of shape `(3, 7, 2, 9)` will result in an output ndarray of shape `(42, 9)`. More examples are given in the function signature.
```
def special_reshape(array): 
	original_shape = array.shape 
	new_first_dimension = np.prod(original_shape[:-1]) 
	new_second_dimension = original_shape[-1] 
	reshaped_array = array.reshape(new_first_dimension, new_second_dimension) 
	return reshaped_array
```


(b) Implement linear, which takes in an input $1-D$ ndarray (which we will call vector from now on) $x$, weight matrix $W$, and bias vector $b$. Perform a linear transformation on x using $W$ and $b$ using the formula $y = Wx + b$.
```
def linear(x: np.ndarray, W: np.ndarray, b: np.ndarray):
	return W@x + b
```


(c) Implement `sigmoid`, which takes in an input vector and performs the sigmoid operation on each element. The output ndarray should have the same shape as the input ndarray. Recall that the sigmoid function on a scalar input is:
$$
\displaylines{
\sigma(x)=\frac{1}{1+e^{-x}}
}
$$
```
def sigmoid(x: ndarray):
	return (1 + np.exp(-1*x))**(-1)
```


(d) Implement `two_layer_nn`, which simulates the forward-propagation of a two-layer neural network, given the weight matrices and bias vectors for layers 1 ($W1$ and $b1$) and 2 ($W2$ and $b2$) and an input vector $x$. For this neural network, you should perform a linear transformation AND sigmoid activation after both layers. Note that you must use linear and sigmoid in your implementation for this question.

```
def two_layer_nn(W1: ndarray, b1: ndarray, W2: ndarray, b2: ndarray, x: ndarray):
	x_after_first_layer = sigmoid(linear(x, W1, b1))
	x_final = sigmoid(linear(x_after_first_layer, W2, b2))
	return x_final
```


# Problem 6

Let $f(\mu,\Sigma)$ be the probability density function of a normally distributed random variable in $\mathbb{R}^{2\times2}$

(b) $\mu=\left[\begin{matrix} 1\\1 \end{matrix}\right]$, $\Sigma=\left[\begin{matrix} 1&0\\0&2 \end{matrix}\right]$
```
import matplotlib.pyplot as plt
import numpy as np

def plot_func(mu: np.ndarray, sigma: np.ndarray):
  c_max = 1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
  c_min = 0.1*c_max
  c = np.linspace(c_min, c_max, 10)
  x = np.linspace(-10, 10, 100)
  y = np.linspace(-10, 10, 100)
  X, Y = np.meshgrid(x, y)
  Z = np.zeros(X.shape)
  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      Z[i, j] = c_max * np.exp(-0.5 * np.array([X[i, j], Y[i, j]] - mu).T @ np.linalg.inv(sigma) @ np.array([X[i, j], Y[i, j]] - mu))
      
  plt.contour(X, Y, Z, c)
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.title('Contour plot of the Gaussian distribution')
  plt.show()
```

```
if __name__ == '__main__':

  mu = np.array([1, 1])

  sigma = np.array([[1, 0], [0, 2]])

  plot_func(mu, sigma)
```


(c) $\mu=\left[\begin{matrix} -1\\2 \end{matrix}\right]$, $\Sigma=\left[\begin{matrix} 2&1\\1&4 \end{matrix}\right]$
```
if __name__ == '__main__':

  mu = np.array([-1, 2])

  sigma = np.array([[2, 1], [1, 4]])

  plot_func(mu, sigma)
```


# Problem 7
No data :(
***