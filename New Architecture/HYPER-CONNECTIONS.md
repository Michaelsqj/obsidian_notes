![[images/Pasted image 20250119115744.png]]
![[images/Pasted image 20250115162632.png]]
![[images/Pasted image 20250119111540.png]]
# Motivation
Pre-Norm leads to collapse in deep representation
Post-Norm leads to gradient vanishing
Predefined connection strength between input output

Adjusting the connection strength between different layers

==Analysis of increased computational cost due to replicated h==

# Methods

## Static hyper connections

- Hyper hidden matrix
  $$H=(h_1,h_2,...,h_n)^T$$
- Hyper connection matrix
  $$HC=\begin{pmatrix}0&B\\A_m&A_r\end{pmatrix}$$
	- Depth connections
	  $$DC=\begin{pmatrix}B\\diag(A_r)\end{pmatrix}=\begin{pmatrix}\beta_1&\beta_1&...&\beta_n\\\alpha_{1,1}&\alpha_{2,2}&...&\alpha_{n,n}\end{pmatrix}$$
	
	- Width connections
	  $$WC=\begin{pmatrix}A_m&A_r\end{pmatrix}$$
- 

## Data dependent hyper connections

The parameters for connection strength $A_m,A_r,B$ are dependent on the input matrix $H$ (eg., via a linear layer)

$$HC(H)=\begin{pmatrix}0&B(H)\\A_m(H)&A_r(H)\end{pmatrix}$$


# Overall algorithm
1. Expand the initial hidden vector $h^0$ to $H^0$
2. In layer $k$, 
	1. the input is $H^{k-1}$ 
	2. Using $A_m$, $A_r$ to transform $H^{k-1}$ to two parts, $$h'\in R^d$$and $$H'\in R^{n\times d}$$
	4. **Only** $h'$ passes through the original layer computation (e.g., attention/ convolution/ FFN etc.), and then transformed by $B^k$ to get $\hat{H}$
	5. $H^k=\hat{H}+H'$

