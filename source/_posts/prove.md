---
title: 任意两个n维随机向量的夹角服从什么分布？
abbrlink: 55f9
date: 2025-05-27 19:56:30
tags:
hidden: true
---

当考虑两个 $n$ 维随机向量的夹角分布时，通常需要对这些随机向量的分布做出假设。最常见和可处理的情况是假设这两个随机向量是独立同分布（i.i.d.）的，并且每个分量服从标准正态分布，即它们是独立的高斯随机向量。

### 假设和定义

设 $\mathbf{X}$ 和 $\mathbf{Y}$ 是两个独立的 $n$ 维随机向量，且每个分量服从标准正态分布，即：
$\mathbf{X} = (X\_1, X\_2, \dots, X\_n)^T$，其中 $X\_i \sim \mathcal{N}(0, 1)$ 且相互独立。
$\mathbf{Y} = (Y\_1, Y\_2, \dots, Y\_n)^T$，其中 $Y\_i \sim \mathcal{N}(0, 1)$ 且相互独立。
此外，$\mathbf{X}$ 和 $\mathbf{Y}$ 之间也是独立的。

两个向量 $\mathbf{X}$ 和 $\mathbf{Y}$ 之间的夹角 $\theta$ 定义为：
$$\cos \theta = \frac{\mathbf{X} \cdot \mathbf{Y}}{\|\mathbf{X}\| \|\mathbf{Y}\|}$$
其中 $\mathbf{X} \cdot \mathbf{Y} = \sum\_{i=1}^n X\_i Y\_i$ 是点积，$\|\mathbf{X}\| = \sqrt{\sum\_{i=1}^n X\_i^2}$ 是欧几里得范数。
由于 $\mathbf{X}$ 和 $\mathbf{Y}$ 是随机向量，它们的夹角 $\theta$ 也是一个随机变量，其取值范围在 $[0, \pi]$ 之间。

### 夹角 $\theta$ 的分布

在这种假设下，两个独立高斯随机向量的夹角 $\theta$ 的分布有一个著名的结果，通常通过以下步骤推导：

1.  **标准化和旋转不变性：**
    高斯随机向量的球对称性是一个关键性质。如果 $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, I\_n)$，那么任何正交变换 $Q\mathbf{Z}$ 仍然服从 $\mathcal{N}(\mathbf{0}, I\_n)$。
    我们可以将 $\mathbf{Y}$ 的方向固定在某个方向（例如第一个坐标轴）而不失一般性，因为夹角只依赖于相对方向。更严谨的做法是，利用旋转不变性，我们可以假设 $\mathbf{Y}$ 落在某个特定方向上。

2.  **利用球坐标系或投影：**
    考虑将 $\mathbf{X}$ 投影到由 $\mathbf{Y}$ 定义的一维子空间上。
    令 $\mathbf{u}\_{\mathbf{Y}} = \frac{\mathbf{Y}}{\|\mathbf{Y}\|}$ 为 $\mathbf{Y}$ 方向上的单位向量。
    那么 $\mathbf{X} \cdot \mathbf{Y} = \mathbf{X} \cdot (\|\mathbf{Y}\| \mathbf{u}\_{\mathbf{Y}}) = \|\mathbf{Y}\| (\mathbf{X} \cdot \mathbf{u}\_{\mathbf{Y}})$.
    所以 $\cos \theta = \frac{\mathbf{X} \cdot \mathbf{u}\_{\mathbf{Y}}}{\|\mathbf{X}\|}$.

    由于 $\mathbf{X}$ 和 $\mathbf{Y}$ 是独立的，并且由于高斯分布的旋转不变性，我们可以先固定 $\mathbf{Y}$ 的方向（例如，假设 $\mathbf{Y}$ 沿 $e\_1$ 轴，即 $\mathbf{Y} = (\|\mathbf{Y}\|, 0, \dots, 0)^T$）。在这种情况下：
    $\cos \theta = \frac{X\_1 \|\mathbf{Y}\|}{\|\mathbf{X}\| \|\mathbf{Y}\|} = \frac{X\_1}{\|\mathbf{X}\|}$.
    这种简化的想法是正确的，但需要更正式的证明。

    更正式的思路是，我们可以将 $\mathbf{X}$ 分解为平行于 $\mathbf{Y}$ 的分量和垂直于 $\mathbf{Y}$ 的分量。
    令 $\mathbf{P}\_{\mathbf{Y}} = \frac{\mathbf{Y}\mathbf{Y}^T}{\|\mathbf{Y}\|^2}$ 为到 $\mathbf{Y}$ 张成的子空间的投影矩阵。
    $\mathbf{X} = (\mathbf{X} \cdot \mathbf{u}\_{\mathbf{Y}})\mathbf{u}\_{\mathbf{Y}} + (\mathbf{X} - (\mathbf{X} \cdot \mathbf{u}\_{\mathbf{Y}})\mathbf{u}\_{\mathbf{Y}})$.
    设 $S\_1 = \mathbf{X} \cdot \mathbf{u}\_{\mathbf{Y}}$ (投影到 $\mathbf{Y}$ 方向的长度) 和 $S\_2 = \|\mathbf{X} - (\mathbf{X} \cdot \mathbf{u}\_{\mathbf{Y}})\mathbf{u}\_{\mathbf{Y}}\|$ (垂直于 $\mathbf{Y}$ 方向的长度)。
    那么 $\|\mathbf{X}\|^2 = S\_1^2 + S\_2^2$ (根据勾股定理，因为两个分量正交)。
    我们有 $\cos \theta = \frac{S\_1}{\|\mathbf{X}\|} = \frac{S\_1}{\sqrt{S\_1^2 + S\_2^2}}$.

    关键的结论是：如果 $\mathbf{X} \sim \mathcal{N}(\mathbf{0}, I\_n)$，并且 $\mathbf{Y}$ 是另一个独立的向量，那么：
    * $\mathbf{X} \cdot \mathbf{u}\_{\mathbf{Y}}$ (一个标量) 服从 $\mathcal{N}(0, 1)$ 分布。
    * $\|\mathbf{X} - (\mathbf{X} \cdot \mathbf{u}\_{\mathbf{Y}})\mathbf{u}\_{\mathbf{Y}}\|^2$ (剩余 $(n-1)$ 维空间中的平方范数) 服从自由度为 $(n-1)$ 的卡方分布 $\chi^2\_{n-1}$，并且与 $\mathbf{X} \cdot \mathbf{u}\_{\mathbf{Y}}$ 独立。

    因此，令 $Z = \mathbf{X} \cdot \mathbf{u}\_{\mathbf{Y}} \sim \mathcal{N}(0, 1)$，并且 $W = \|\mathbf{X} - (\mathbf{X} \cdot \mathbf{u}\_{\mathbf{Y}})\mathbf{u}\_{\mathbf{Y}}\|^2 \sim \chi^2\_{n-1}$。
    那么 $\cos \theta = \frac{Z}{\sqrt{Z^2 + W}}$.

3.  **推导 $\cos \theta$ 的分布：**
    令 $T = \frac{Z}{\sqrt{W/(n-1)}}$ 服从自由度为 $n-1$ 的学生t分布。
    这个和我们的 $\cos \theta$ 形式略有不同。

    我们考虑 $V = \cos^2 \theta = \frac{Z^2}{Z^2 + W}$.
    由于 $Z^2 \sim \chi^2\_1$ 和 $W \sim \chi^2\_{n-1}$ 且两者独立，那么 $V$ 服从 $\text{Beta}(\alpha, \beta)$ 分布。
    具体来说，如果 $U\_1 \sim \chi^2\_{\nu\_1}$ 和 $U\_2 \sim \chi^2\_{\nu\_2}$ 独立，则 $\frac{U\_1}{U\_1 + U\_2} \sim \text{Beta}(\nu\_1/2, \nu\_2/2)$.
    所以 $\cos^2 \theta \sim \text{Beta}(1/2, (n-1)/2)$.

    这是关于 $\cos^2 \theta$ 的分布。现在我们需要找到 $\theta$ 的分布。
    令 $X = \cos^2 \theta$. 其概率密度函数 (PDF) 为：
    $$f\_X(x) = \frac{1}{B(1/2, (n-1)/2)} x^{1/2-1} (1-x)^{(n-1)/2-1} \quad \text{for } x \in (0, 1)$$
    其中 $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ 是 Beta 函数。

    我们希望找到 $\theta$ 的 PDF $f\_{\theta}(\theta)$.
    由于 $\theta \in [0, \pi]$，$\cos \theta \in [-1, 1]$，但 $\cos^2 \theta \in [0, 1]$。
    由于 $\mathbf{X}$ 和 $\mathbf{Y}$ 的分量是独立高斯随机变量，它们的夹角是对称的，即 $\theta$ 和 $\pi - \theta$ 发生的概率是相同的。因此，我们通常考虑 $\theta \in [0, \pi/2]$ 的情况（即 $\cos \theta \ge 0$），或者说我们关心的是非负的夹角。
    如果 $\mathbf{X}$ 和 $\mathbf{Y}$ 都是随机的，那么 $\cos \theta$ 既可以为正也可以为负。但是由于 $\mathbf{X}$ 和 $\mathbf{Y}$ 都是中心化的，它们的夹角分布是对称的，即 $P(\theta \le \alpha) = P(\theta \ge \pi-\alpha)$。

    对于 $\theta \in [0, \pi]$，我们有 $x = \cos^2 \theta$. 那么 $\frac{dx}{d\theta} = -2 \cos \theta \sin \theta$.
    $$f\_{\theta}(\theta) = f\_X(\cos^2 \theta) \left| \frac{dx}{d\theta} \right| = \frac{1}{B(1/2, (n-1)/2)} (\cos^2 \theta)^{-1/2} (1-\cos^2 \theta)^{(n-1)/2-1} | -2 \cos \theta \sin \theta |$$
    $$f\_{\theta}(\theta) = \frac{1}{B(1/2, (n-1)/2)} \frac{1}{|\cos \theta|} (\sin^2 \theta)^{(n-3)/2} (2 |\cos \theta| |\sin \theta|)$$
    $$f\_{\theta}(\theta) = \frac{2}{B(1/2, (n-1)/2)} (\sin \theta)^{n-2} \quad \text{for } \theta \in [0, \pi]$$
    这个结果是在 $\cos \theta$ 既可以是正也可以是负的情况下导出的，所以 $\sin \theta$ 总是非负的。
    注意：对于 $\theta = 0$ 或 $\theta = \pi$， $\sin \theta = 0$，此时 PDF 为 0。这也很合理，因为恰好平行或反平行的概率为 0。

    这个分布是关于 $\theta$ 的 PDF。
    对于 $n=2$ (二维情况)，$f\_{\theta}(\theta) = \frac{2}{B(1/2, 1/2)} (\sin \theta)^{0} = \frac{2}{\pi}$ for $\theta \in [0, \pi]$.
    这意味着在二维平面上，两个独立高斯随机向量的夹角服从 $[0, \pi]$ 上的均匀分布。这是一个非常直观的结果，因为在二维中，方向是均匀分布的。

    对于 $n=3$ (三维情况)，$f\_{\theta}(\theta) = \frac{2}{B(1/2, 1)} (\sin \theta)^{1} = \frac{2}{\Gamma(1/2)\Gamma(1)/\Gamma(3/2)} \sin \theta = \frac{2}{\sqrt{\pi} \cdot 1 / (\sqrt{\pi}/2)} \sin \theta = \sin \theta$ for $\theta \in [0, \pi]$.
    在三维空间中，两个独立高斯随机向量的夹角分布是 $\sin \theta$，这与在球面上随机取两个点，它们之间的测地线距离（对应夹角）的分布是一致的。

### 总结

如果两个 $n$ 维随机向量 $\mathbf{X}$ 和 $\mathbf{Y}$ 是独立的，并且每个分量服从标准正态分布（即 $\mathbf{X}, \mathbf{Y} \sim \mathcal{N}(\mathbf{0}, I\_n)$），那么它们之间的夹角 $\theta$ 的概率密度函数 (PDF) 为：
$$f\_{\theta}(\theta) = \frac{2}{B(1/2, (n-1)/2)} (\sin \theta)^{n-2}, \quad \text{for } \theta \in [0, \pi]$$
其中 $B(a, b)$ 是 Beta 函数，$\Gamma(z)$ 是 Gamma 函数。

**特殊情况：**
* **$n=2$：** $f\_{\theta}(\theta) = \frac{2}{B(1/2, 1/2)} (\sin \theta)^{0} = \frac{2}{\pi}$, for $\theta \in [0, \pi]$。即均匀分布。
* **$n=3$：** $f\_{\theta}(\theta) = \frac{2}{B(1/2, 1)} (\sin \theta)^{1} = \sin \theta$, for $\theta \in [0, \pi]$。

### 证明细节

上述推导中一些关键点的证明：

**1. 高斯向量的旋转不变性**
如果 $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, I\_n)$，且 $Q$ 是一个正交矩阵，那么 $Q\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, I\_n)$。
这是因为 $E[Q\mathbf{Z}] = Q E[\mathbf{Z}] = \mathbf{0}$，协方差矩阵 $Cov(Q\mathbf{Z}) = E[(Q\mathbf{Z})(Q\mathbf{Z})^T] = E[Q\mathbf{Z}\mathbf{Z}^T Q^T] = Q E[\mathbf{Z}\mathbf{Z}^T] Q^T = Q I\_n Q^T = Q Q^T = I\_n$.

**2. $\mathbf{X} \cdot \mathbf{u}\_{\mathbf{Y}}$ 和 $\|\mathbf{X} - (\mathbf{X} \cdot \mathbf{u}\_{\mathbf{Y}})\mathbf{u}\_{\mathbf{Y}}\|^2$ 的分布及独立性**
由于 $\mathbf{X}$ 和 $\mathbf{Y}$ 独立，我们可以条件化在 $\mathbf{Y}$ 上。
固定一个单位向量 $\mathbf{u}$ (代表 $\mathbf{u}\_{\mathbf{Y}}$)。
考虑 $\mathbf{X}$ 的分解：$\mathbf{X} = (\mathbf{X} \cdot \mathbf{u})\mathbf{u} + (\mathbf{X} - (\mathbf{X} \cdot \mathbf{u})\mathbf{u})$.
令 $Z = \mathbf{X} \cdot \mathbf{u}$. $Z$ 是 $n$ 个独立标准正态变量的线性组合 ($Z = \sum X\_i u\_i$)，且 $\sum u\_i^2 = 1$.
$E[Z] = \sum E[X\_i] u\_i = 0$.
$Var(Z) = \sum Var(X\_i) u\_i^2 = \sum u\_i^2 = 1$.
因此 $Z \sim \mathcal{N}(0, 1)$.

现在考虑 $W = \|\mathbf{X} - (\mathbf{X} \cdot \mathbf{u})\mathbf{u}\|^2$.
这是一个 $n$ 维向量 $\mathbf{X}$ 投影到与 $\mathbf{u}$ 正交的 $(n-1)$ 维空间后的范数的平方。
我们可以构造一个正交矩阵 $Q$ 使得其第一列是 $\mathbf{u}$。
那么 $Q^T \mathbf{X}$ 仍然服从 $\mathcal{N}(\mathbf{0}, I\_n)$。
$\mathbf{X} \cdot \mathbf{u} = (Q^T \mathbf{X})\_1$ (第一个分量)。
$\|\mathbf{X} - (\mathbf{X} \cdot \mathbf{u})\mathbf{u}\|^2 = \|Q^T \mathbf{X} - (Q^T \mathbf{X})\_1 Q^T \mathbf{u}\|^2$.
由于 $Q^T \mathbf{u} = (1, 0, \dots, 0)^T$，所以
$\|\mathbf{X} - (\mathbf{X} \cdot \mathbf{u})\mathbf{u}\|^2 = \|(Q^T \mathbf{X})\_1 (1,0,\dots,0)^T + \sum\_{j=2}^n (Q^T \mathbf{X})\_j e\_j - (Q^T \mathbf{X})\_1 (1,0,\dots,0)^T\|^2$
$= \|\sum\_{j=2}^n (Q^T \mathbf{X})\_j e\_j\|^2 = \sum\_{j=2}^n (Q^T \mathbf{X})\_j^2$.
由于 $(Q^T \mathbf{X})\_j \sim \mathcal{N}(0, 1)$ 且独立，所以 $\sum\_{j=2}^n (Q^T \mathbf{X})\_j^2 \sim \chi^2\_{n-1}$.
并且 $(Q^T \mathbf{X})\_1$ 与 $\sum\_{j=2}^n (Q^T \mathbf{X})\_j^2$ 独立。
因此 $Z = (Q^T \mathbf{X})\_1 \sim \mathcal{N}(0, 1)$ 和 $W = \sum\_{j=2}^n (Q^T \mathbf{X})\_j^2 \sim \chi^2\_{n-1}$ 独立。

**3. 从 $\cos^2 \theta$ 的 Beta 分布到 $\theta$ 的 PDF**
这个推导是标准的变量变换。
$X = \cos^2 \theta$ 的 PDF 是 $f\_X(x) = \frac{1}{B(1/2, (n-1)/2)} x^{-1/2} (1-x)^{(n-3)/2}$.
令 $\theta \in [0, \pi]$.
由于 $\theta \mapsto \cos^2 \theta$ 不是单射，我们需要小心处理。
但注意到 $f\_{\theta}(\theta)$ 中的 $(\sin \theta)^{n-2}$ 项，当 $\theta \in [0, \pi]$ 时，$\sin \theta \ge 0$.
$|dx/d\theta| = |-2 \cos \theta \sin \theta| = 2 |\cos \theta| |\sin \theta|$.
当 $\theta \in [0, \pi/2]$ 时，$\cos \theta \ge 0$, $\sin \theta \ge 0$. $f\_{\theta}(\theta) = f\_X(\cos^2 \theta) (2 \cos \theta \sin \theta)$.
当 $\theta \in [\pi/2, \pi]$ 时，$\cos \theta \le 0$, $\sin \theta \ge 0$. $f\_{\theta}(\theta) = f\_X(\cos^2 \theta) (-2 \cos \theta \sin \theta)$.
综合起来：
$f\_{\theta}(\theta) = \frac{1}{B(1/2, (n-1)/2)} (\cos^2 \theta)^{-1/2} (\sin^2 \theta)^{(n-3)/2} (2 |\cos \theta| \sin \theta)$
$= \frac{1}{B(1/2, (n-1)/2)} \frac{1}{|\cos \theta|} (\sin \theta)^{n-3} (2 |\cos \theta| \sin \theta)$
$= \frac{2}{B(1/2, (n-1)/2)} (\sin \theta)^{n-2}$，对于 $\theta \in [0, \pi]$。

这个结果的物理直观意义是，在高维空间中，随机向量更倾向于相互正交（夹角接近 $\pi/2$）。这是因为随着维度的增加，与某个特定方向正交的子空间所占的“体积”越来越大。例如，对于 $n=3$，夹角分布是 $\sin \theta$，在 $\theta = \pi/2$ 处达到峰值。对于更高的 $n$，$(\sin \theta)^{n-2}$ 会在 $\pi/2$ 处变得更尖锐。

### 扩展情况

* **如果向量不是独立同分布的，或者不是高斯的？**
    在这种情况下，夹角分布会变得非常复杂，通常没有简单的解析形式。例如，如果分量服从均匀分布，或者向量之间存在相关性，那么推导将非常困难。
* **如果向量不是中心化的（均值不为零）？**
    如果 $E[\mathbf{X}] \neq \mathbf{0}$ 或 $E[\mathbf{Y}] \neq \mathbf{0}$，那么夹角分布也会改变。通常我们会先对向量进行中心化处理，或者需要考虑更复杂的几何概率模型。

我们这里的讨论严格限制在两个独立且分量服从标准正态分布的 $n$ 维随机向量的夹角。这是最常见且有解析解的情况。