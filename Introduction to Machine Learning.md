# Lecture 1

## Examples of learning problems 
- Recognizing digits
- Classifying spam emails
- Predict the price of a stock 6 month from now
- predict rating of a movie $i$ by a customer $j$
- Determine credit-worthiness for a mortgage or a credit card transaction

## Types of problems
- **Classification**, where the output is a lable from a finate set (example: handwritten digit recognition on MNIST)
- **Regression**, where the output is real-valued
- **Ranking problems**, such as an internet query for a document relevant to a search term

## The machine learning approach to object recognition
- **Training time**
	- Compute **feature vectors** for positive and negative examples of image patches
	- Train a **classifier**
- **Test time**
	- Compute feature vector on image patch
	- Evaluate classifier
### example

![[Pasted image 20240629114043.png]]
In feature space, positive and negative examples are just points
![[Pasted image 20240629114231.png]]
There are mutiple ways to classify a new point
- **Nearst neighbor rule**, whrere "Transfer lable of nearst example"![[Pasted image 20240629114422.png]]![[Pasted image 20240629115212.png]]![[Pasted image 20240629115230.png]]
- Linear classifier rule, where the boundary are discribed as $\overrightarrow{w} \cdot \overrightarrow{x} + \overrightarrow{b} = 0$![[Pasted image 20240629114929.png]]![[Pasted image 20240629115250.png]]**Trick**: ![[Pasted image 20240629120247.png]]![[Pasted image 20240629120324.png]]
- Multilayer perceptrons a.k.a **Neural Networks**

## Neural Networks
### Single layer neural network![[Pasted image 20240629120546.png]]
### Two layer neural network![[Pasted image 20240629120511.png]]
## Train a neural network
- **Goal**: Find $\overrightarrow{w}$ such that $\overrightarrow{O}$ is as close as possible to $\overrightarrow{y}$ (desired output)
- **Approach**: 
	- Define **Loss function $L(\overrightarrow{w})$**
	- Compute $\triangledown L$
	- $w_{new} = w_{old} - \eta\triangledown L$ (**gradient descent**)

## Training a single layer neural network
- A good choice of loss function is the **cross entropy**: $L = -\sum\limits_{input data}(y_{i}ln(O_{i})+(1-y_{i})ln(1-O_{i})$
- we model the activation function $g$ as a **sigmid**: $g(Z) = \frac{1}{1+e^{-z}}$
- Fiding $w$ reduces to **logistic regression**!

## Training a two or more layer neural network
- We compute the gradient with respect to all the weights: from input to hidden layer, and hidden layer to output layer
- We can use stochastic gradient descent as before. The loss function is no longer convex, so we can only find local minima. That may be good enough for many applications
- The complexity of computing the gradient in the naïve version is quadratic in the number of weights. The back- propagation algorithm is a trick that enables it to be computed in linear time
- We can add a regularization term to penalize large weights; that usually improves the performance

## How do we choose a classifier
- If we knew the true probability distribution of the features conditioned on the classes, there is a “correct” answer – the Bayes classifier. This minimizes the probability of misclassification![[Pasted image 20240629123853.png]]

## Two kinds of error
- **training set error**: we trian a classifier to minimize training set error
- **test set error**: At run time, we will take the trained classifier and use it to classify previously unseen examples. The error on these is called test set error

## Validation and Cross-Validation
- If the test set error is much greater than training set error, that is called **over-fitting**
- To avoid over-fitting, we can measure error on a held-out set of training data, called the **validation set**
- We could divide the data into $k$-folds, use $k-1$ of these to train and test on the remaining fold. This is **cross-validation**
***
# Lecture 2

*maximum likelihood estimation*

## ML: main abstract concept
- Training data set: *the $y$-label provides supervision*
	- $D=\{(x_{i},y_{i})\}^{N}_{i}$,  $x_{i}\in R^{D}$, *supervised*
		- $y\in\{-1,1\}$ *classification*
		- $y\in R$ *regression*
	- $D=\{(x_{i})\}_{i=1}^{N}$ *unsupervised*
- Model class/ hypothesis class: $f(\mathbf{x}|\mathbf{w},\mathbf{b})=\mathbf{w}^{T}\mathbf{x}+\mathbf{b}$ *linear models*
- Optimization goal: find "good" values of parameters ($\mathbf{w}, \mathbf{b}$)
- to define "good", we have:
	- Loss function: $L(a,b)=(a-b)^{2}$
	- Learning objective: $argmin_{w,b}\sum\limits_{i=1}^{N}L(y_{i},f(\mathbf{x},\mathbf{w},\mathbf{b}))$ *optimization problem*

## Maximum Likelihood Estimation (MLE)
This principle gives a usful, principled and widely-used loss function to estimate to estimate parameters of statistical models (from linear regression, to neural networks, and beyond).

## Reminder: Rrobability distributions
Random varible (**RV**) is a function: $x\rightarrow\mathbb{R}$
There are two types of RV
- **Discrete RV**, e.g. coin toss heads/tails
- **Continuous RV**, e.g. height
Discrete RVs have **Probability Mass Function (PMF)**![[Pasted image 20240630162238.png]]
Continuous RVs have **Probability Density Function (PDF)**![[Pasted image 20240630162301.png]]
## The basic set-up of MLE
- Given data $D=\{x_{i}\}^{N}_{i=1}$ for $x_{i}\in\mathbb{R}^{d}$
- Assume a set (family) of distributions on $\mathbb{R}^{d}$, $\{p_{\theta}(\mathbf{x})|\theta\in\Theta\}$
the $\theta$ defines the mean ($\mu$) and the variance ($\sigma^{2}$) for $\mathbf{x}\in\mathbb{R}^{d}$
- Assume $D$ contains samples from one of these distributions: $x_{i}\sim p_{\hat{\theta}}(x)$
- This assumes that each element of $D$ is identically and independently distributed (iid)
- **Goal**: "learn"/estimate the value of $\theta$ that defines the distribution from which the data came
- **Definition**: $\theta_{MLE}$ is a MLE for $\theta$ with respect to the data and the set of diatributions, if $\theta_{MLE}=argmax p(D|\theta)$

Note that $p(D|\theta)=p(\{x_{i}\}^{N}_{i=1}|\theta)=\Pi_{i=1}^{N}p(x_{i}|\theta)$
Some properties of MLE
- The MLE is a consistent estimator: meaning that as we get more and more data (drawn from one distribution in our family), then we converge to estimating the true value of $\theta$ for $D$
- The MLE is statistically efficient: it's making good use of the data available to it
- The value of $p(D|\theta_{MLE})$ is invariant to re-parameterization
- MLE can still yeild a parameter estimate even when the data were not generated from that family (phew&caveat emptor)

## Example of MLE for univariate Gaussian
**Goal:** $argumax p(D|\theta)$ from set of data $D=\{x_{i}\}^{N}_{i=1}$
- Assume data are generated as $X\sim N(x|\mu,\sigma^{2}) = \frac{e^{-\frac{(x-\mu)^{2}}{2\sigma^{2}}}}{\sqrt{2\pi\sigma^{2}}}$
- So assume MLE family of distributions, $p(X=x|\theta)=N(X|\mu,\sigma^{2})$
- First step, write down the likelihood function:
	- $p(D|\theta)=p(x_{1},x_{2},...x_{N}|\mu,\sigma^{2}) = \Pi^{N}_{i=1}p(x_{i}|\mu,\sigma^{2})$
- The product of the terms is a little inconvenient to work with...
- The *log likelihood (LL)* is a monotonically increasing function of the likelihood.
	- $\log p(D|\theta)=\sum\limits_{i=1}^{N}\log p(x_{i}|\mu,\sigma^{2})$
- Therefore $\theta_{MLE}=argmax p(D|\theta) = argmax \log p(D|\theta)$
- How to solve the optimization problem above?
- Fiding a setting of the parameters for which the derivatives are 0 (a stationary point)
- Then check whether the setting is a maximum (negative second derivative), a minimum (positive second derivative)
- if parameters >1, check if Hessian is negative definite; for 1D Gaussian, Hessian is diagonal, so can check each separately.
- $$
\displaylines{
\sum\limits_{i=1}^{N}\log p(x_{i}|\mu,\sigma^{2}) \\
= \sum\limits_{i=1}^{N}\left[\log\frac{1}{\sqrt{2\pi\sigma^{2}}}-\frac{(x_{i}-\mu)^{2}}{2\sigma^{2}}\right]\\
\therefore \frac{\partial \sum\limits_{i=1}^{N}\log p(x_{i}|\mu,\sigma^{2})}{\partial \mu} = \frac{\sum\limits_{i=1}^{N}x_{i}-N\mu}{\sigma^{2}} \\
\frac{\partial \sum\limits_{i=1}^{N}\log p(x_{i}|\mu,\sigma^{2})}{\partial \sigma} = -\frac{N}{\sigma}+\frac{\sum\limits_{i=1}^{N}(x_{i}-\mu)^{2}}{\sigma^{3}}\\
\therefore \begin{cases} \mu = \frac{\sum\limits_{i=1}^{N}x_{i}}{N}\\ 
\sigma^{2} = \frac{\sum\limits_{i=1}^{N}(x_{i}-\mu)^{2}}{N}
\end{cases}
}
$$
## MLE yeilds a "point estimate" of our parameter
- When performing MLE, we get just one single estimate of the parameter, $\theta$, rather than a distribution over it which captures uncertainty
- In Bayesian statistics, we obtain a (posterior) distruibution over $\theta$. 

## Example of MLE for the mutinomial distribution
- Consider a six-sided die that we will roll, and we want to know the probablity of each side of the die turning up ($\theta = \theta_{1}, \theta_{2}, \theta_{3}, \theta_{4}, \theta_{5}, \theta_{6}$)
- Assume we have observed $N$ rolls, with RV, $X\sim p_{\theta}(X)$
- We write that $P(X=k|\theta)=\theta_{k}$
- First we have $1=\sum\limits_{i=1}^{6}\theta_{i}$
- Then the likelihood function: $\Pi_{i=1}^{N}p_{\theta}(x_{i})=\Pi_{i=1}^{N}\Pi^{6}\theta_{k}^{\sum 1[x_{i}=k]}=\Pi_{k=1}^{6}\theta_{k}^{n_{k}}$
- $\theta_{MLE} = argmax\sum_\limits{k=1}^{6}\log\theta_{k}^{n_{k}}$
- Use lagrange multipliers to solve this optimization problem: $L(\theta,\lambda) = \log p(D|\theta)+\lambda(1-\sum\limits_{i=1}^{6}\theta_{i})$
- $\frac{\partial L(\theta,\lambda)}{\theta_{k}}=\frac{n_{k}}{\theta_{k}}-\lambda$
- $\therefore \theta_{k}=\frac{n_{k}}{\lambda}$
- since we have $\sum\limits_{i=1}^{6}\theta_{i}=\sum\limits_{i=1}^{6}\frac{n_{K}}{\lambda}=\frac{\sum\limits_{i=1}^{6}n_{k}}{\lambda}=\frac{N}{\lambda}=1\rightarrow \lambda=N$
- All together, $\theta_{k} = \frac{n_{k}}{N}$
## Relationship between likelihood, cross-entropy, etc.
As mentioned in Lecture 1 ![[Introduction to Machine Learning#Training a single layer neural network]]
A good choice of loss function is cross entropy rather than MLE, but acturally, they are equivalent.
- The *cross-entropy* is a term from *information theory*
- To fully understand the connection between MLE and maximizing the cross-entropy, we need to know some concepts from information theory
	- Entropy
	- Cross-entropy
	- KL-divergence (relative entropy)

### Entropy: a measure of expected surprise
*Think about flipping a coin once, and how surprised you would be at observing a head?*
- The "surprise" of observing that a distance random variable $Y$ takes on value $k$ is: $\log\frac{1}{P(Y=K)}$
- As $P(Y=k)\rightarrow 0$, the surprise of oberving $k$ approaches $\infty$
- As $P(Y=k)\rightarrow 1$, the surprise of oberving $k$ approaches $0$
- The entropy of the distribution $Y$ is the expect surprise: $H(Y)\equiv-\sum P(Y=K)\log P(Y=K)$
### Entropy of a RV $Y$
- "**High Entropy**"
	- $Y$ is from a uniform like distribution
	- Flat histogram
	- Values sampled from it are less predictable
- "**Low Entropy**"
	- $Y$ is from a varied (peaks and valleys) distribution
	- Histogram has many lows and highs
	- Values sampled from it are more predictable

### From Relative Entropy to Cross-entrop 
*then to MLE!*
$D_{KL}(P||Q)=\sum P(x)log\frac{P(x)}{Q(x)}$
$=E_{p(x)}[\log\frac{1}{Q(x)}]-E_{p(x)}[\log\frac{1}{P(x)}]$
$=H(P,Q)-H(P)$
the $H(P,Q)$ here is **Cross-Entropy** while $D_{KL}(P||Q)$ being **Relative Entropy**
Now, if we want to minimize the Cross-entropy, which is
$argmin_{\theta}D_{KL}(\hat{p}||p(x|\theta))=argmin_{\theta}H(\hat{p}||p(x|\theta))-H(\hat{p})$
$=argmaxE_{\hat{p}_{data}}[\log p(x|\theta)]=argmax\sum\limits_{i=1}^{N}log p(x|\theta)$
which is exactly the MLE problem

## Extra
in the example above![[Introduction to Machine Learning#Example of MLE for the mutinomial distribution]]
We haven't check if this stationary point a maximum or a minmum, instead of checking the Hessian, we could also check the relative-entropy to see if it is minmum:
$D_{KL}(p_{data}||p(x|\theta))=\sum\limits_{k=1}^{6}P_{data}(X=k)\log\frac{P_{data}(X=K)}{P(X=k|\theta)}=\sum\limits_{k=1}^{6}P_{data}(X=k)\log\frac{\frac{n_{k}}{N}}{\theta_{k}}=\sum\limits_{k=1}^{6}P_{data}(X=k)\log 1=0$
Obvious the relative-entropy reaches minmun.
***
# Lecture 3
*Mutivariate Gaussians (MVG)*
## Mutivariate Gaussians (MVG)
The pdf of a univariate Gaussian (normal) distribution is:
$p(x,\mu,\sigma^{2})=\frac{e^{-\frac{1}{2\sigma^{2}(x-\mu)^{2}}}}{\sqrt{2\pi\sigma^{2}}}$
Then  we have the mutivariate extension of this is for $x\in\mathbb{R}^{d}, u\in\mathbb{R}^{d}$ and $\Sigma\in\mathbb{R}^{d\times d}$ and PSD
$p(x,\mu,\Sigma)=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}\exp\left(-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)\right)$

## Why having a lecture on MVGs
*MVGs permeat much of classical and mordern day ML*

- Classification: generative vs. discriminative
- Unsupervised models: Principle Components Analysis & autoencoders
- Advanced topics: Gaussian Process Regression (and deep versions thereof), etc

## MVG distributions
- Consider two quantities, height and weight (of humans)
- Given the arguments of CLT (Central Limit Theorem) with genetics, it’s plausible that each of these is Gaussian distributed, so lets assume:$$X_{h}\sim N(\mu_{h},\sigma^{2}_{h}), X_{w}\sim N(\mu_{w},\sigma^{2}_{w})$$
Suppose I want the joint distribution, $p([X_{h}=x_{h},X_{w}=x_{w}])$, how would we write it down?
![[Pasted image 20240702093911.png]]
- Each point is a sample from some 2D pdf, $p([x_{h},x_{w}])$
- If we computed the mean of this diatruibution, $u=[\mu_{1}, \mu_{2}]$, it would be $[\mu_{h}, \mu_{w}]$
- How do we compute/wirte the "spread" of the points?
- Can we use $p([x_{h}, x_{w}])=N(x_{h},\mu_{h},\sigma^{2}_{h})*N(x_{w}, \mu_{w}, \sigma^{2}_{w})$ ?
![[Pasted image 20240702094648.png]]
For independent RVs, $p([x_{1},x_{2}])=N(x_{1},\mu_{1},\sigma^{2}_{1})*N(x_{2}, \mu_{2}, \sigma^{2}_{2})$
Then if we rotate the coordinate system to be "axis aligned", we could have $p([x_{h}, x_{w}])=N(x_{1},\mu_{1},\sigma^{2}_{1})*N(x_{2}, \mu_{2}, \sigma^{2}_{2})$
![[Pasted image 20240702095016.png]]
- But how do we do a rotation?
- Multiply by an appropriate **orthonormal matrix** $Q$: $[x_{1}, x_{2}]^{T}=Q[x_{h}, x_{w}]^{T}$

## "Baby" case: variables are independent, and each is 1D
$X\sim N(\mu_{x}, \sigma^{2}_{x})$, $Y\sim N(\mu_{y},\sigma^{2}_{y})$
Then we have
$p([x,y])=p(x, \mu_{x},\sigma^{2}_{x})*p(y,\mu_{y},\sigma^{2}_{y})$
$=\frac{e^{-\frac{1}{2\sigma^{2}_{x}(x-\mu_{x})^{2}}}}{\sqrt{2\pi\sigma^{2}_{x}}}*\frac{e^{-\frac{1}{2\sigma^{2}_{y}(y-\mu_{y})^{2}}}}{\sqrt{2\pi\sigma^{2}_{y}}}$
$=\frac{1}{2\pi\sigma_{x}\sigma_{y}}\exp\left(-\frac{1}{2\sigma^{2}_{x}(x-\mu_{x})^{2}}-\frac{1}{2\sigma^{2}_{y}(y-\mu_{y})^{2}}\right)$
$=\frac{1}{2\pi\sigma_{x}\sigma_{y}}\exp\left(-\frac{1}{2\sigma^{2}_{x}(x-\mu_{x})^{2}}-\frac{1}{2}\left([x-\mu_{x}, y-\mu_{y}]\left[ \begin{matrix} \sigma^{2}_{x}&0\\0&\sigma^{2}_{y}\end{matrix} \right]^{-1} \left[\begin{matrix}x-\mu_{x} \\ y-\mu_{y}\end{matrix}\right] \right) \right)$  (because $\left[ \begin{matrix} \sigma^{2}_{x}&0\\0&\sigma^{2}_{y}\end{matrix} \right]^{-1}=\left[ \begin{matrix} \frac{1}{\sigma^{2}_{x}}&0\\0&\ \frac{1}{\sigma^{2}_{y}}\end{matrix} \right]$)

$=\frac{1}{2\pi\sigma_{x}\sigma_{y}}\exp\left(-\frac{1}{2\sigma^{2}_{x}(x-\mu_{x})^{2}}-\frac{1}{2}\left([x-\mu_{x}, y-\mu_{y}]\Sigma^{-1} \left[\begin{matrix}x-\mu_{x} \\ y-\mu_{y}\end{matrix}\right] \right) \right)$
where $\Sigma$ is called **covariance** matrix in the MVG

## Review: Expectations, Variance and Covariance
### Expectation
$E(X)$
- For discrete RVs, $E(X)=\sum xp(x)$
- For continus RVs, $E(X)=\int xp(x)$
**Properties of Expections**
- Linearity: $E(3A+2B) = 3E(A)+2E(B)$
- Let $X_{1}, ...,X_{n}$ be independent RVs, then: $E(\Pi_{i=1}^{n}X_{i})=\Pi_{i=1}^{n}E(X_{i})$
- $E(X+C) = E(X)+C$

### Variance
Let $X$ be a RV with expection $\mu=E(X)$
then **Variance** can be defined as $E(X-\mu)^{2}$
**Properties of Variance**
- $Var(X)=E(X^{2})-\mu^{2}$
- $Var(aX+b) = a^{2}Var(X)$
- Let $X_{1}, ..., X_{n}$ be independent RVs, then: $Var(\sum\alpha_{i}X_{i})=\sum Var(\alpha_{i}X_{i})$

### Covariance
$Cov(X,Y) = E((X-E(X))(Y-E(Y)))=E((X-\mu_{x})(Y-\mu_{y}))=E(XY)-E(X)E(Y)$
$Cov(X,X)=Var(X,X)$, for independent RVs, $Cov(X,Y)=0$

### Correlation
$Corr(X,Y) = \frac{Cov(X,Y)}{\sqrt{Var(X)Var(Y)}}$

## Covariance Matrix
$\Sigma\in\mathbb{R}^{n\times n}$
$\Sigma_{ij}=Cov(X_{i},X_{j})$, for input RVs:$[X_{1}, X_{2},...,X_{n}]^{T}$
In the previous example, the input is $[X,Y]^{T}$, then we have
$\Sigma\in\mathbb{R}^{2\times2}$, $\Sigma_{11}=Cov(X,X)=Var(X,X)=\sigma_{x}^{2}, \Sigma_{12}=\Sigma_{21}=Cov(X,Y)=0, \Sigma_{22}=Cov(Y,Y)=\sigma_{y}^{2}$
$\therefore \Sigma=\left[\begin{matrix} \sigma^{2}_{x}&0\\0&\sigma^{2}_{y} \end{matrix}\right]$
$Cov(X_{i}, X_{j})=0$ iff (if and only if) $X_{i}$ and $X_{j}$ are independent

## From baby case to the general case
*How can we better understand the general case, with $X\in\mathbb{R}^{d}$ and non-independence between the components?*
$p(x,\mu,\Sigma)=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}\exp\left(-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)\right)$
**The MVG is at its core a quadratic form**, MVG has two main terms
- **Quadratic term**, where most of the "actions happens".
- **Normalizing Constant**, which ensures that the distribution integrates to **$1$**

### Quadratic term
*level set (contour line) of MVG pdf comprise the values $x$ for which $p(x)$ is a constant*

## Sphering a MVG
- To "sphere" a MVG is to alter it so as make all its contour lines be spheres (also called "whitening")
- (useful for manipulation of MVGs related to PCA, advanced linear regressions, etc.)![[Pasted image 20240702193231.png]](decorrelated data also called diagonalized data)

## Linear Algebra: Diagonalizing a matrix
- For MVG, the covariance matrix (and its inverse) is symmetric and positive semidefinite (PSD)
- Symmetric because $Cov(X,Y)=Cov(Y,X)$
To Diagonal a matrix, we could work as below
- $\Sigma$, is symmetric and PSD
- $\text{then } \Sigma=QDQ^{T}$, in which D contains real eigenvalues, $Q$ is an orthonormal matrix ($QQ^{T}=I$)
- therefore $Q$ contains orthonormal vectors which implicates independent RVs: $Q=[q_{1}, ..., q_{n}]$ each is length 1 and any two are orthogonal
- because $Q^{T}\Sigma Q=D$, then $q_{i}^{T}\Sigma q_{i}=\lambda_{i}\rightarrow q_{i}^{T}\Sigma q_{i} q_{i}^{T}=\lambda_{i}q_{i}^{T}$
- since the length of $q_{i}$=1, then $q_{i}^{T}q_{i}=1$, therefore $q_{i}^{T}\Sigma=q_{i}^{T}\lambda_{i}$, which indecates that each row of $Q$ is a orthogonal eigenvector of each eigenvalue
- **example**
	- $\left[\begin{matrix}x&y\end{matrix}\right]\left[\begin{matrix}5&4\\4&5\end{matrix}\right]\left[\begin{matrix}x\\y\end{matrix}\right]$
	- $\Sigma=\left[\begin{matrix}5&4\\4&5\end{matrix}\right]$ is PSD
	- $D=\left[\begin{matrix}9&0\\0&1\end{matrix}\right]$
	- eigenvetors: $\frac{1}{\sqrt{2}}\left[\begin{matrix}1\\1\end{matrix}\right]$, $\frac{1}{\sqrt{2}}\left[\begin{matrix}1\\-1\end{matrix}\right]$ (each for 9 and 1)
	- then $Q=\frac{1}{\sqrt{2}}\left[\begin{matrix}1&1\\1&-1\end{matrix}\right]$

## Geometric intuition: "de-sphering" a MVG
- Let $X\sim N(0,I)$
- Let $\Sigma=QDQ^{T}$ be a covariace matrix factored into its eigenvalues and diagonal matrix. Can also write it as $\Sigma=(QD^{\frac{1}{2}})(D^{\frac{1}{2}})Q^{T}=AA^{T}$
- Let $Y=AX+\mu$, then $Y\sim N(\mu,\Sigma)$
Can decompose any MVG in terms of a "Scaling" ($D$), "Rotation" ($Q$), and "Shift" ($\mu$)
***

