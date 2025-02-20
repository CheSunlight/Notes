# Bayesian Solutions for the Factor Zoo: We Just Ran Two Quadrillion Models

## 论文信息

### 作者

Svetlana Bryzgalova, Jiantao Huang, and Christian Julliard

Bryzgalova 是伦敦商学院的助理教授，Huang 是香港大学的助理教授，Julliard 是伦敦政治经济学院（LSE）的副教授。前两位之前博士都就读于 LSE。

### 收录情况

JF2023

## 解决什么问题

在高维设定（factor zoo）下解决弱因子（weak factors）识别问题与过拟合问题，同时给出一个不管在样本内还是样本外定价能力都强的 SDF。

## 前置知识

### 频率派估计 SDF

#### 线性形式的 SDF

假设有 $K$ 个因子 $\bm{f}_{\! t} = (f_{1t},\ f_{2t},\ \cdots,\ f_{Kt})^{\top},\ t=1,\ 2,\ \cdots,\ T$ 和 $N$ 个 test assets（多空组合）$\bm{R}_t = (R_{1t},\ R_{2t},\ \cdots,\ R_{Nt})^{\top}$，我们通常用因子来构造线性形式的 SDF $m_t$ 来满足 test assets 的定价条件：

$$
\begin{align}
&m_t = 1 - [\bm{f}_{\! t} - \E(\bm{f}_{\! t})]^{\top} \bm{\lambda}_{\bm{f}} \label{1} \\
&\text{s.t.}\quad \E(m_t \bm{R}_t) = \bm{0}_{N} \label{2}
\end{align}
$$

其中 $\bm{\lambda}_{\bm{f}}$ 为 SDF 载荷，也被称作因子的[风险价格](asset_pricing/prices_of_risk_and_risk_premia.md)（prices of risk）。

把 $\eqref{1}$ 式代入限制条件 $\eqref{2}$ 可以解得

$$
\begin{align}
\bm{\mu}_{\bm{R}} = \bm{C}_{\! \bm{f}} \bm{\lambda}_{\bm{f}} \label{3}\\
\end{align}
$$

其中 $\bm{\mu}_{\bm{R}} := \E(\bm{R}_t)$，$\bm{C}_{\! \bm{f}}$ 为 $\bm{R}_t$ 与 $\bm{f}_{\! t}$ 之间的协方差矩阵。

> [!NOTE|label:注意]
> 以下 $\mu_{x}$ 均代表某一变量 $x$ 的期望，且默认消除时间维度；$\bar{x}$ 则代表 $x$ 的样本均值。

在允许存在定价误差的情况下，依据 $\eqref{3}$ 式，**因子的风险价格可以通过以下截面回归得到**：

$$
\begin{equation}
\bm{\mu}_{\bm{R}} = \lambda_c \bm{1}_{N} + \bm{C}_{\! \bm{f}} \bm{\lambda}_{\bm{f}} + \bm{\alpha} = \bm{C} \bm{\lambda} + \bm{\alpha} \label{4}
\end{equation}
$$

其中 $\bm{C} := (\bm{1}_{N},\ \bm{C}_{\! \bm{f}})$，$\bm{\lambda}^{\top} := (\lambda_c,\ \bm{\lambda}_{\bm{f}}^{\top})$。$\lambda_c$ 为平均定价误差（average mispricing），$\bm{\alpha}$ 为特质误差，即每个 test assets 在平均定价误差外的定价误差。从回归的角度来看，**实际上 $\lambda_c$ 是截距项，$\bm{\alpha}$ 是误差项。**

#### GMM

上述截面回归的解通常可以用 <strong>GMM（Generalized Method of Moments，广义矩估计）</strong>得到。假设 $\E(\bm{\alpha}) = \bm{0}_{N}$，我们有如下矩条件：

$$
\E[\bm{g}_t(\lambda_c,\ \bm{\lambda}_{\bm{f}},\ \bm{\mu}_{\bm{f}})] = \E \begin{bmatrix} \bm{R}_t - \lambda_c \bm{1}_{N} - \bm{R}_t (\bm{f}_{\! t} - \bm{\mu}_{\bm{f}})^{\top} \bm{\lambda}_{\bm{f}} \\ \bm{f}_{\! t} - \bm{\mu}_{\bm{f}} \\\end{bmatrix} = \begin{bmatrix}	\bm{0}_{N} \\	\bm{0}_{K} \\\end{bmatrix}
$$

可以看到参数的数量为 $2 K + 1$，而矩条件的数量是 $N + K$，当 $N > K + 1$ 时，矩条件数量大于参数数量，我们基本上得不到准确的解。GMM 为了处理这样的过度识别（overidentification）问题，引入了矩条件的权重，对更应该限制的条件给予更大的权重。因此 GMM 的目标函数为

$$
\begin{equation}
\underset{\lambda_c,\ \bm{\lambda}_{\bm{f}},\ \bm{\mu}_{\bm{f}}}{\min} ~ \bm{g}_{T}(\lambda_c,\ \bm{\lambda}_{\bm{f}},\ \bm{\mu}_{\bm{f}})^{\top} \bm{W} \bm{g}_{T}(\lambda_c,\ \bm{\lambda}_{\bm{f}},\ \bm{\mu}_{\bm{f}}) \label{5}
\end{equation}
$$

其中 $\bm{g}_{T}(\lambda_c,\ \bm{\lambda}_{\bm{f}},\ \bm{\mu}_{\bm{f}}) := \frac{1}{T} \sum\limits_{t=1}^{T} \bm{g}_t(\lambda_c,\ \bm{\lambda}_{\bm{f}},\ \bm{\mu}_{\bm{f}})$，$\bm{W}$ 控制矩条件的权重。

> [!TIP|label:提示]
> 使用权重矩阵会让某些条件权重变小，极端情况就是变为 0，相当于减少了条件数量，因此能改善过度识别问题。

#### GLS

对于 OLS 和 **GLS（Generalized Least Squares，广义最小二乘）**，权重矩阵分别为

$$
\bm{W}_{\text{OLS}} = \begin{bmatrix}	\bm{I}_{\! N} & \bm{0}_{N \times K} \\	\bm{0}_{K \times N} & \kappa \bm{I}_{\! K} \\\end{bmatrix},\quad
\bm{W}_{\text{GLS}} = \begin{bmatrix}	\bm{\Sigma}_{\bm{R}}^{-1} & \bm{0}_{N \times K} \\	\bm{0}_{K \times N} & \kappa \bm{I}_{\! K} \\\end{bmatrix}
$$

其中 $\kappa > 0$ 是一个很大的常数来保证 $\bm{\widehat{\mu}}_{\bm{f}} = \frac{1}{T} \sum\limits_{t=1}^{T} \bm{f}_{\! t}$，$\bm{\Sigma}_{\bm{R}}$ 是收益率的协方差矩阵。

由 GMM 目标函数 $\eqref{5}$ 解出来的估计分别为

$$
\bm{\widehat{\lambda}}_{\text{OLS}} = \left(\bm{\widehat{C}}^{\top} \bm{\widehat{C}} \right)^{-1} \bm{\widehat{C}}^{\top} \bm{\overline{R}},\quad \bm{\widehat{\lambda}}_{\text{GLS}} = \left(\bm{\widehat{C}}^{\top} \bm{\Sigma}_{\bm{R}}^{-1} \bm{\widehat{C}} \right)^{-1} \bm{\widehat{C}}^{\top} \bm{\Sigma}_{\bm{R}}^{-1} \bm{\overline{R}}
$$

<strong>不同的权重矩阵从回归的角度来看其实是不同的误差分布假设。</strong>对于 OLS 来说，我们假设误差服从 iid 的正态分布，也就是 $\bm{\alpha} \sim \mathcal{N}(\bm{0}_{N},\ \sigma^{2}\bm{I}_{\! N})$；而 **GLS 则是考虑了截面上误差的相关性，即假设 $\bm{\alpha} \sim \mathcal{N}(\bm{0}_{N},\ \sigma^{2} \bm{\Sigma}_{\bm{R}})$。**

> [!NOTE|label:注意]
> 对于标准的 GLS，我们用的权重矩阵为误差的协方差矩阵，通常用 OLS 得到的残差进行估计。本文中直接使用 $\bm{\Sigma}_{\bm{R}}$ 作为误差的协方差矩阵是因为假设 $\eqref{4}$ 式这个模型是正确的，那么对于每个时间点 $t$，我们有
>
> $$ \bm{R}_{t} = \bm{C} \bm{\lambda} + \bm{\varepsilon}_{t},\quad \bm{\varepsilon}_{t} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(\bm{0}_{N},\ \bm{\Sigma}_{\bm{R}}) $$
> 
> 而截面误差 $\bm{\alpha} = \frac{1}{T}\sum_{t=1}^{T} \bm{\varepsilon}_{t}$，根据中心极限定理，$\bm{\alpha} \mid \bm{\Sigma}_{\bm{R}} \sim \mathcal{N}(\bm{0}_{N},\ \frac{1}{T} \bm{\Sigma}_{\bm{R}})$。然而在允许定价误差存在的情况下，定价误差的程度由观测到的数据来决定，因此允许协方差矩阵可以被 scaling 是比较好的，于是 GLS 的误差假设变成 $\bm{\alpha} \sim \mathcal{N}(\bm{0}_{N},\ \sigma^{2} \bm{\Sigma}_{\bm{R}})$。由于所有模型都会有一定程度上的误设，我们期望 $\sigma^{2} > \frac{1}{T}$。

## 创新点

假设在 $K$ 个因子中有 $K_1$ 个可交易的因子和 $K_2$ 个不可交易的因子，记 $\bm{f}_{\! t} = \left(\bm{f}_{\! t}^{(1) \top},\ \bm{f}_{\! t}^{(2) \top} \right) ^{\top}$，$\bm{f}_{\! t}^{(1) \top}$ 为可交易的因子，$\bm{f}_{\! t}^{(2) \top}$ 为不可交易的因子。令 $\bm{Y}_{\! t}$ 表示因子和 test assets 收益率的并，由于可交易的因子其实也可以作为 test assets 的收益率，我们不用重复列出，因此 $\bm{Y}_{\! t}^{\top} := \left(\bm{R}_t^{\top},\ \bm{f}_{\! t}^{(2) \top} \right) ^{\top}$ 是一个 $N + K_2$ 维的向量。

> [!TIP|label:提示]
> 将可交易的因子作为 test assets 的收益率，说明这些可交易的因子都是超额收益率。

### 使用分层贝叶斯估计因子的风险溢价

本文涉及的变量之间的关系可由如下概率图所描述：

![](drawio/bayesian_solutions_for_the_factor_zoo.png)

其中被橙色填充的变量 $\bm{Y}$ 为真实观测到的数据，标橙的 $\bm{\color{#e3620c}{\lambda}}$ 是我们所关心的变量，左边的圆角矩形代表 $\bm{\mu}_{\bm{Y}}$ 由 $\bm{\mu}_{\bm{f}^{(2)}}$ 和 $\bm{\mu}_{\bm{R}}$ 组成，右边同理。

#### 时序贝叶斯

已有的观测到的变量只有时序变量 $\bm{Y}$，所以我们先对它进行建模：假设 test assets 的收益率和因子服从时间上 iid 的多元正态分布，即 $\bm{Y}_{\! t} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(\bm{\mu}_{\bm{Y}},\ \bm{\Sigma}_{\bm{Y}})$，由此可以固定时序似然函数为

$$
p(\bm{Y} \mid \bm{\mu}_{\bm{Y}},\ \bm{\Sigma}_{\bm{Y}}) \propto \left\vert\bm{\Sigma}_{\bm{Y}}\right\vert^{-\frac{T}{2}} \exp \left\{-\frac{1}{2} \operatorname{tr}\left[\bm{\Sigma}_{\bm{Y}}^{-1} \sum_{t=1}^T \left(\bm{Y}_{\! t} - \bm{\mu}_{\bm{Y}}\right) \left(\bm{Y}_{\! t} - \bm{\mu}_{\bm{Y}}\right)^{\top}\right]\right\}
$$

从贝叶斯的角度，我们需要对参数有一些先验假设，因此作者使用了一个 NIW（Normal Inverse Wishart）的扩散先验（diffuse prior）$\pi(\bm{\mu}_{\bm{Y}},\ \bm{\Sigma}_{\bm{Y}}) \propto \left\vert \bm{\Sigma}_{\bm{Y}} \right\vert^{-\frac{p+1}{2}} $，也就是均值先验为均匀分布，方差先验为逆 Wishart 分布。

> [!TIP|label:提示]
> 扩散先验又被称为无信息先验（uninformative prior），指分布比较平坦的先验。均匀分布是最简单的扩散先验，NIW 也可以作为扩散先验。
> 
> 不对方差使用均匀先验是因为方差矩阵本身就具有结构&mdash;对称且元素非负，Wishart 分布正是定义在对称且非负定的随机矩阵上，而逆 Wishart 分布又是正态分布关于方差的[共轭分布](/papers/shrinking_the_cross-section.md#先验分布、后验分布与共轭分布)，因此对方差采用逆 Wishart 分布的先验。

在这个先验下的参数后验为

$$
\bm{\mu}_{\bm{Y}} \mid \bm{\Sigma}_{\bm{Y}},\ \bm{Y} \sim \mathcal{N}(\bm{\widehat{\mu}}_{\bm{Y}},\ \bm{\Sigma}_{\bm{Y}} / T)
$$

$$
\bm{\Sigma}_{\bm{Y}} \mid \bm{Y} \sim \mathcal{W}^{-1} \left(T-1,\ \sum\limits_{t=1}^{T} (\bm{Y}_{\! t} - \bm{\widehat{\mu}}_{\bm{Y}}) (\bm{Y}_{\! t} - \bm{\widehat{\mu}}_{\bm{Y}})^{\top} \right)
$$

其中 $\bm{\widehat{\mu}}_{\bm{Y}} := \frac{1}{T} \sum\limits_{t=1}^{T} \bm{Y}_{\! t}$，$\mathcal{W}^{-1}$ 为逆 Wishart 分布。

通过时序贝叶斯，我们可以根据观测到的 $\bm{Y}_{\! t}$ 去构建 $\bm{\Sigma}_{\bm{Y}}$ 的后验分布，从而从分布中抽样得到 $\bm{\Sigma}_{\bm{Y}}$，而抽样得到 $\bm{\Sigma}_{\bm{Y}}$ 后我们又可以得到 $\bm{\mu}_{\bm{Y}}$ 的后验分布，从而从分布中抽样得到 $\bm{\mu}_{\bm{Y}}$。于是 $\bm{\mu}_{\bm{Y}}$ 和 $\bm{\Sigma}_{\bm{Y}}$ 都可以“观测”到了：

![](drawio/bayesian_solutions_for_the_factor_zoo1.png)

#### 截面贝叶斯

有了截面变量 $\bm{\mu}_{\bm{Y}}$ 和 $\bm{\Sigma}_{\bm{Y}}$，我们可以对它们进行建模（这里我们只对它们的一部分——$\bm{\mu}_{\bm{R}}$ 和 $\bm{C}_{\! \bm{f}}$ 进行建模）：由 $\eqref{4}$ 式我们知道，$\bm{\mu}_{\bm{R}}$ 和 $\bm{C}$ 是与风险价格 $\bm{\lambda}$ 以及误差 $\bm{\alpha}$ 有关的。在 OLS 的误差假设下，$\bm{\alpha} \sim \mathcal{N}(\bm{0}_{N},\ \sigma^{2}\bm{I}_{\! N})$，据此我们可以写出截面的似然函数：

$$
p(\bm{\mu}_{\bm{R}},\ \bm{C}\mid \bm{\lambda},\ \sigma^{2}) = (2 \pi \sigma^{2})^{-\frac{N}{2}} \exp \left\{- \frac{1}{2 \sigma^{2}} (\bm{\mu}_{\bm{R}} - \bm{C} \bm{\lambda})^{\top} (\bm{\mu}_{\bm{R}} - \bm{C} \bm{\lambda}) \right\} 
$$

同样地，我们可以对参数 $\bm{\lambda}$ 和 $\sigma^{2}$ 进行先验假设：$\pi(\bm{\lambda},\ \sigma^{2}) \propto \sigma^{-2}$，则参数的后验为

$$
\bm{\lambda} \mid \sigma^2,\ \bm{\mu}_{\bm{R}},\ \bm{C} \sim \mathcal{N}\left(\underbrace{(\bm{C}^{\top} \bm{C})^{-1} \bm{C}^{\top} \bm{\mu}_{\bm{R}}}_{\bm{\widehat{\lambda}}},\ \underbrace{\sigma^2 (\bm{C}^{\top} \bm{C})^{-1}}_{\bm{\Sigma}_{\bm{\lambda}}}\right)
$$

$$
\sigma^{2} \mid \bm{\mu}_{\bm{R}},\ \bm{C} \sim \mathcal{IG} \left(\frac{N - K - 1}{2},\ \frac{(\bm{\widehat{\mu}}_{\bm{R}} - \bm{C} \bm{\widehat{\lambda}})^{\top} (\bm{\widehat{\mu}}_{\bm{R}} - \bm{C} \bm{\widehat{\lambda}})}{2} \right) 
$$

其中 $\mathcal{IG}$ 代表逆 Gamma 分布。

> [!TIP|label:提示]
> Wishart 分布是 Gamma 分布在多元情况下的 generalization，这里 $\sigma^{2}$ 是一元的，因此我们用的是 NIG（Normal Inverse Gamma）的先验。

> [!TIP|label:假定模型是正确的]
> 如果假定模型是正确的，也就是 $\bm{\alpha} = \bm{0}_{N}$，那么令 $\sigma^{2} \to 0$，风险价格 $\bm{\lambda}$ 的后验就是一个在 $(\bm{C}^{\top}\bm{C})^{-1}\bm{C}^{\top}\bm{\mu}_{\bm{R}}$ 处的 Dirac 分布，也就是说 $\bm{\lambda}$ 是一个常数向量。每一次对 $\bm{\mu}_{\bm{Y}}$ 和 $\bm{\Sigma}_{\bm{Y}}$ 抽样我们都可以得到一个 $\bm{\lambda}$，抽多了我们就可以得到 $\bm{\lambda}$ 的置信区间。对于弱因子来说，尽管每次抽样会因为 $\bm{c}^{\top}\bm{c}$ 的 near singularity 而让 $\bm{\lambda}$ 变得很大，但由于抽样具有不确定性（不像频率派的确定估计），我们得到的 $\bm{\lambda}$ 可能是正的很大也有可能是负的很大，最终 $\bm{\lambda}$ 的置信区间有很大概率是包含 $\bm{0}$ 的，因此通过贝叶斯估计的方法得到 $\bm{\lambda}$ 能够识别弱因子。
> 
> 除了风险价格 $\bm{\lambda}$，我们还关心截面回归的 $R^{2}$，而通过抽样我们同样可以计算贝叶斯版本的 $R^{2}$：$$ R^{2} = 1 - \frac{(\bm{\mu}_{\bm{R}} - \bm{C} \bm{\lambda})^{\top} (\bm{\mu}_{\bm{R}} - \bm{C} \bm{\lambda})}{(\bm{\mu}_{\bm{R}} - \bar{\mu}_{\bm{R}} \bm{1}_{N})^{\top}(\bm{\mu}_{\bm{R}} - \bar{\mu}_{\bm{R}} \bm{1}_{N})} $$
> 
> 其中 $\bar{\mu}_{\bm{R}} = \frac{1}{N} \sum\limits_{i=1}^{N} \mu_{\bm{R},\ i}$ 为样本 panel 均值，$\mu_{\bm{R},\ i}$ 为 $\bm{\mu}_{\bm{R}}$ 中第 $i$ 个元素。

在 GLS 的误差假设下，即 $\bm{\alpha} \sim \mathcal{N}(\bm{0}_{N},\ \sigma^{2} \bm{\Sigma}_{\bm{R}})$，我们可以得到相似的结果：

$$
p(\bm{\mu}_{\bm{R}},\ \bm{C},\ \bm{\Sigma}_{\bm{R}} \mid \bm{\lambda},\ \sigma^{2}) = (2 \pi \sigma^{2})^{-\frac{N}{2}} \exp \left\{- \frac{1}{2 \sigma^{2}} (\bm{\mu}_{\bm{R}} - \bm{C} \bm{\lambda})^{\top} \bm{\Sigma}_{\bm{R}}^{-1} (\bm{\mu}_{\bm{R}} - \bm{C} \bm{\lambda}) \right\} 
$$

$$
\bm{\lambda} \mid \sigma^2,\ \bm{\mu}_{\bm{R}},\ \bm{C},\ \bm{\Sigma}_{\bm{R}} \sim \mathcal{N}\left(\underbrace{(\bm{C}^{\top} \bm{\Sigma}_{\bm{R}}^{-1} \bm{C})^{-1} \bm{C}^{\top} \bm{\Sigma}_{\bm{R}}^{-1} \bm{\mu}_{\bm{R}}}_{\bm{\widehat{\lambda}}},\ \underbrace{\sigma^2 (\bm{C}^{\top} \bm{\Sigma}_{\bm{R}}^{-1} \bm{C})^{-1}}_{\bm{\Sigma}_{\bm{\lambda}}}\right)
$$

$$
\sigma^{2} \mid \bm{\mu}_{\bm{R}},\ \bm{C},\ \bm{\Sigma}_{\bm{R}} \sim \mathcal{IG} \left(\frac{N - K - 1}{2},\ \frac{(\bm{\widehat{\mu}}_{\bm{R}} - \bm{C} \bm{\widehat{\lambda}})^{\top} \bm{\Sigma}_{\bm{R}}^{-1} (\bm{\widehat{\mu}}_{\bm{R}} - \bm{C} \bm{\widehat{\lambda}})}{2} \right) 
$$

> [!NOTE|label:注意]
> 以下我们将用 $\text{data}$ 来代替 OLS 假设下的 $\bm{\mu}_{\bm{R}},\ \bm{C}$，以及 GLS 假设下的 $\bm{\mu}_{\bm{R}},\ \bm{C},\ \bm{\Sigma}_{\bm{R}}$。尽管它们对于 $\bm{Y}_{\! t}$ 来说是参数，但对于我们最关心的 $\bm{\lambda}$ 来说它们就是可观测的数据。
>
> 由于 GLS 假设下的推导与 OLS 差不多，为了方便，接下来我们只讨论 OLS 假设下的情况，但在实证中用的是 GLS 假设下的估计。

得到了 $\bm{\lambda}$ 和 $\sigma^{2}$ 的后验，我们可以通过 Gibbs sampling 的方法对它们进行交替采样：

1. 对 $\sigma^{2}$ 设定初值；
2. 循环：依次对 $\bm{\Sigma}_{\bm{Y}},\ \bm{\mu}_{\bm{Y}},\ \bm{\lambda},\ \sigma^{2}$ 采样。

> [!TIP|label:提示]
> 在抽样得到 $\bm{\mu}_{\bm{Y}}$ 后，作者做了标准化的操作，即将 $\bm{\mu}_{\bm{Y}}$ 中每一个元素都除以对应的标准差，对于 test assets $\bm{R}$ 来说，这一操作相当于把它们的均值变成了夏普比。在 [Shrinking the cross-section](papers\shrinking_the_cross-section.md#因子风险价格的压缩估计) 中也有类似的操作（对夏普比做岭回归），但在 Shinking the cross-section 中，对 test assets 的均值先验方差为 $\bm{\Sigma}^{2}$，后验方差不为 $\bm{\Sigma}$ 的倍数，因此处理成夏普比时并不等同于做标准化的操作，而是为了让风险价格独立同分布；而本文是均匀先验，后验方差为 $\bm{\Sigma} / T$，因此做标准化恰好可以得到夏普比。

> [!NOTE|label:注意]
> 接下来所有的 $\bm{\mu}_{\bm{Y}}$ 都是经过标准化处理后的。

![](drawio/bayesian_solutions_for_the_factor_zoo2.png)

通过时序贝叶斯和截面贝叶斯，我们已经能够得到 $\bm{\lambda}$ 的置信区间（且对弱因子也有效），但**在 factor zoo 的设定下，包含所有因子的模型不见得是最好的模型，我们希望能够用贝叶斯的方法做变量选择，看哪些因子构成的 SDF 最牛。**

### 使用贝叶斯方法进行变量选择

假定一个二元变量 $\bm{\gamma} = (\gamma_0,\ \gamma_1,\ \cdots,\ \gamma_{K})^{\top}$，当 $\gamma_j = 1$ 表示我们选择因子 $j$ 作为变量，$\gamma_j = 0$ 则表示不选因子 $j$，$\gamma_0 = 1$ 表示我们的模型总是包含截距项。$\bm{\gamma}$ 中的每个元素都服从伯努利分布，即每个变量都有一定概率被选中，而**我们要做的就是通过数据推断出每个变量被选中的概率。**

![](drawio/bayesian_solutions_for_the_factor_zoo3.png)

定义变量的个数为 $p_{\bm{\gamma}} := \sum_{j = 0}^{K} \gamma_j$，$\bm{C}_{\! \bm{\gamma}}$ 代表只包含已选变量的协方差矩阵，$\bm{\lambda}_{\bm{\gamma}}$ 和 $\bm{\lambda}_{-\bm{\gamma}}$ 分别代表只包含已选或未选变量的风险价格。

接下来我们首先阐述对 $\bm{\lambda}$ 使用均匀先验的坏处，然后引入更合适的先验来对 $\bm{\gamma}$ 进行推断。

#### 对风险价格使用均匀先验的坏处

在截面贝叶斯中，尽管 $\bm{\lambda}$ 和 $\sigma^{2}$ 的联合先验为 NIG，但 $\bm{\lambda}$ 的边缘先验是均匀的。**在使用 $\bm{\gamma}$ 进行变量选择时，均匀先验会让弱因子有很大概率被选中。**

当我们使用 $\bm{\gamma}$ 选择某些变量，先验可以写为 $\begin{cases} \pi(\bm{\lambda}_{\bm{\gamma}},\ \sigma^{2}) \propto \sigma^{-2} \\ \bm{\lambda}_{-\bm{\gamma}} = 0 \end{cases}$，则似然函数

$$
\begin{equation}
p(\text{data} \mid \bm{\gamma}) \propto (2 \pi)^{\frac{p_{\bm{\gamma}}}{2}} \textcolor{#e3620c}{\left\vert \bm{C}_{\! \bm{\gamma}}^{\top} \bm{C}_{\! \bm{\gamma}} \right\vert^{-\frac{1}{2}}} \frac{\Gamma(\frac{N - p_{\bm{\gamma}}}{2})}{(\frac{N \widehat{\sigma}_{\bm{\gamma}}^{2}}{2})^{\frac{N - p_{\bm{\gamma}}}{2}}} \label{6}\\
\end{equation}
$$

<details>
<summary>计算细节</summary>

$$
\begin{aligned}
p(\text{data} \mid \bm{\gamma}) &= \iint p(\text{data} \mid \bm{\gamma},\ \bm{\lambda},\ \sigma^{2}) \pi(\bm{\lambda},\ \sigma^{2}\mid \bm{\gamma}) ~ \mathrm{d} \bm{\lambda} ~ \mathrm{d} \sigma^{2} \\
&\propto \iint (\sigma^{2})^{-\frac{N + 2}{2}} e^{-\frac{1}{2 \sigma^{2}}(\bm{\mu}_{\bm{R}} - \bm{C}_{\! \bm{\gamma}} \bm{\lambda}_{\bm{\gamma}})^{\top} (\bm{\mu}_{\bm{R}} - \bm{C}_{\! \bm{\gamma}} \bm{\lambda}_{\bm{\gamma}})} ~ \mathrm{d} \bm{\lambda} ~ \mathrm{d} \sigma^{2} \\
&= \iint (\sigma^{2})^{-\frac{N + 2}{2}} e^{-\frac{1}{2 \sigma^{2}}\left[(\bm{\mu}_{\bm{R}} - \bm{C}_{\! \bm{\gamma}} \bm{\widehat{\lambda}}_{\bm{\gamma}})^{\top} (\bm{\mu}_{\bm{R}} - \bm{C}_{\! \bm{\gamma}} \bm{\widehat{\lambda}}_{\bm{\gamma}}) + (\bm{C}_{\! \bm{\gamma}} \bm{\lambda}_{\bm{\gamma}} - \bm{C}_{\! \bm{\gamma}} \bm{\widehat{\lambda}}_{\bm{\gamma}})^{\top} (\bm{C}_{\! \bm{\gamma}} \bm{\lambda}_{\bm{\gamma}} - \bm{C}_{\! \bm{\gamma}} \bm{\widehat{\lambda}}_{\bm{\gamma}}) \right]} ~ \mathrm{d} \bm{\lambda} ~ \mathrm{d} \sigma^{2} \\
&= \iint (\sigma^{2})^{-\frac{N + 2}{2}} e^{-\frac{N \widehat{\sigma}_{\bm{\gamma}}^{2}}{2 \sigma^{2}}} e^{-\frac{( \bm{\lambda}_{\bm{\gamma}} - \bm{\widehat{\lambda}}_{\bm{\gamma}})^{\top} \bm{C}_{\! \bm{\gamma}}^{\top} \bm{C}_{\! \bm{\gamma}} (\bm{\lambda}_{\bm{\gamma}} - \bm{\widehat{\lambda}}_{\bm{\gamma}})}{2 \sigma^{2}}} ~ \mathrm{d} \bm{\lambda} ~ \mathrm{d} \sigma^{2} \\
&= (2 \pi)^{\frac{p_{\bm{\gamma}}}{2}} \left\vert \bm{C}_{\! \bm{\gamma}}^{\top} \bm{C}_{\! \bm{\gamma}} \right\vert^{-\frac{1}{2}}  \int (\sigma^{2})^{-\frac{N - p_{\bm{\gamma}} + 2}{2}} e^{-\frac{N \widehat{\sigma}_{\bm{\gamma}}^{2}}{2 \sigma^{2}}} ~ \mathrm{d} \sigma^{2} \\
&= (2 \pi)^{\frac{p_{\bm{\gamma}}}{2}} \left\vert \bm{C}_{\! \bm{\gamma}}^{\top} \bm{C}_{\! \bm{\gamma}} \right\vert^{-\frac{1}{2}} \frac{\Gamma(\frac{N - p_{\bm{\gamma}}}{2})}{(\frac{N \widehat{\sigma}_{\bm{\gamma}}^{2}}{2})^{\frac{N - p_{\bm{\gamma}}}{2}}} \\
\end{aligned}
$$

其中 $\bm{\widehat{\lambda}}_{\bm{\gamma}} := (\bm{C}_{\! \bm{\gamma}}^{\top} \bm{C}_{\! \bm{\gamma}})^{-1} \bm{C}_{\! \bm{\gamma}}^{\top} \bm{\mu}_{\bm{R}}$，$\widehat{\sigma}_{\bm{\gamma}}^{2} := \frac{(\bm{\mu}_{\bm{R}} - \bm{C}_{\! \bm{\gamma}} \bm{\widehat{\lambda}}_{\bm{\gamma}})^{\top} (\bm{\mu}_{\bm{R}} - \bm{C}_{\! \bm{\gamma}} \bm{\widehat{\lambda}}_{\bm{\gamma}})}{N}$，$\Gamma$ 代表 Gamma 函数。
</details>

因此在这个设定下，如果我们使用 $\bm{\gamma}$ 选择的变量中包含弱因子，那么由于 $\bm{C}_{\! \bm{\gamma}}^{\top} \bm{C}_{\! \bm{\gamma}}$ 的 near singularity，它的行列式趋于 $0$，导致 $\eqref{6}$ 趋于无穷，即似然函数趋于 $1$。根据贝叶斯定理，

$$
p(\bm{\gamma} \mid \text{data}) = \frac{p(\text{data} \mid \bm{\gamma})}{p(\text{data} \mid \bm{\gamma}) + p(\text{data} \mid \bm{\gamma}') + \cdots}
$$

分母上为所有变量选择的可能性下似然函数的加和。由于 $p(\text{data} \mid \bm{\gamma})$ 趋于 $1$，$p(\bm{\gamma} \mid \text{data})$ 也会趋于 $1$，这就导致根据数据，我们倾向于将弱因子包括在变量选择中。

#### 更合适的先验：钉板先验

接下来我们将对风险价格 $\bm{\lambda}$ 采用一个更合适的先验：**钉板先验（spike-and-slab prior）**，来避免对弱因子的错误选择。

<div align='center'>

![](image/2023-03-31-14-49-57.png)
</div align='center'>

如上图所示，<strong>钉板先验由两个均值为 0，方差不同的正态分布组成：黑色的像一根钉子，被称为钉先验（spike），红色的像一块板，被称为板先验（slab）。</strong>钉板先验被广泛应用于涉及变量选择的回归问题中，**未被选择的变量服从钉先验，被选择的变量则服从板先验**，通过这样的设定我们能够推断出 $\bm{\gamma}$ 的后验概率，从而选择最可能的那一组变量。

> [!TIP|label:提示]
> 理论上钉先验应该是一个在 0 处的 Dirac 分布，但实际操作中为了分布的连续性我们通常使用一个方差极小的正态分布。

对于板先验，不同的因子的风险价格应当服从不同方差的正态分布。**因子的风险溢价越高，也就是因子与 test assets 之间的相关性越高，因子具有高的风险价格的概率应该越大，也就是我们需要让它服从更高方差的正态分布**，因此我们对 $\bm{\lambda}$ 的先验可以写成如下形式：

$$
\begin{equation}
\lambda_j \mid \gamma_j,\ \sigma^{2} \sim \mathcal{N}(0,\ r(\gamma_j) \psi_j \sigma^{2}) \label{7}
\end{equation}
$$

其中

$$
\begin{equation}
\begin{split}
r(\gamma_j) = \begin{cases}
    1,\ \gamma_j = 1 \\
    0.001,\ \gamma_j = 0 \\
\end{cases}
\end{split}
\end{equation}
$$

区分钉板先验，

$$
\begin{equation}
\psi_j = \psi \times \bm{\tilde{\rho}}_{j}^{\top} \bm{\tilde{\rho}}_{j}
\end{equation}
$$

决定了不同因子的方差大小， $\psi$ 是一个超参数，$\bm{\tilde{\rho}}_{j} = \bm{\rho}_{j} - (\frac{1}{N} \sum_{i=1}^{N} \rho_{j,\ i}) \cdot \bm{1}_{N}$ 则是因子 $j$ 与所有 test assets 之间的相关系数（demean 后）。**当 $\bm{\gamma}$ 错误包含了弱因子，这样的先验假设会让该弱因子的板先验趋近钉先验（$\bm{\tilde{\rho}}_{j}^{\top} \bm{\tilde{\rho}}_{j} \to 0 \implies \sigma^{2}\psi_j \to 0$），从而达到识别弱因子的效果。**

> 弱因子的定义就是相关性弱，但相关性弱只能代表风险溢价为 $0$，不代表风险价格为 $0$，是否有影响？

> [!TIP|label:提示]
> 将相关系数进行 demean 处理是为了解决水平因子（level factor）的问题。
>
> **有两种因子会导致 $\bm{C}_{\! \bm{\gamma}}^{\top} \bm{C}_{\! \bm{\gamma}}$ 的 near singularity**：一种是**弱因子**，弱因子和所有 test assets 都几乎没有相关性，也就是 $\bm{C}_{\! \bm{\gamma}}$ 对应的那一列都是 0，导致 $\bm{C}_{\! \bm{\gamma}}$ 不满秩；一种是**水平因子**，水平因子与所有 test assets 之间的相关性都很接近，也就是 $\bm{C}_{\! \bm{\gamma}}$ 中对应的那一列值都相同，这会导致这一列与截距那一列（全是 1）成比例，即 $\bm{C}_{\! \bm{\gamma}}$ 依旧不满秩。

记对角线元素为 $(r(\gamma_1)\psi_1)^{-1},\ (r(\gamma_2)\psi_2)^{-1},\ \cdots (r(\gamma_{K})\psi_{K})^{-1}$ 的矩阵为 $\bm{D}$。则上述先验可以表达成以下矩阵形式：

$$
\bm{\lambda} \mid \sigma^{2},\ \bm{\gamma} \sim \mathcal{N}(\bm{0},\ \sigma^{2} \bm{D}^{-1})
$$

#### 引入 ω 辅助 γ 抽样

如果不对 $\bm{\gamma}$ 采样，想要找到最好的变量选择，我们需要考虑每一种可能性。假设有 $30$ 个变量，变量选择的可能性数量为 $2^{30}$，每一种可能性下我们都要做若干次 Gibbs sampling，这样一来计算量是极其大的。

因此为了能够对 $\bm{\gamma}$ 采样，我们对 $\bm{\gamma}$ 有如下先验假设：

$$
\begin{equation}
\pi(\gamma_j = 1 \mid \omega_j) = \omega_j,\quad \omega_j \sim \text{Beta}(a_{\bm{\omega}},\ b_{\bm{\omega}})
\end{equation}
$$

其中 $a_{\bm{\omega}}$ 和 $b_{\bm{\omega}}$ 分别为 Beta 分布的两个参数。

新引入的变量 $\bm{\omega}$ 不仅能够让我们对 $\bm{\gamma}$ 采样，同时还加入了我们对模型 sparsity 的先验。**根据 Beta 分布的特性，$\E(\omega_j) = \frac{a_{\omega}}{a_{\omega} + b_{\omega}}$，即一个因子 $j$ 被选中的期望概率为 $\frac{a_{\omega}}{a_{\omega} + b_{\omega}}$，比如选取 $a_{\omega} = b_{\omega} = 1$ 代表每个因子都有 $\frac{1}{2}$ 的概率被选中；选取 $a_{\omega} = 1$ 和 $b_{\omega} \gg 1$ 则能让模型更为 sparse。**

> [!TIP|label:提示]
> $\text{Beta}(1,\ 1)$ 与 $\text{Uniform}(0,\ 1)$ 是等价的，但是 **Beta 分布是伯努利分布的共轭分布**，对于服从伯努利分布的 $\bm{\gamma}$ 来说，对 $\bm{\omega}$ 选择 Beta 分布是合适且易于更新的。

#### 后验推断

目前为止，我们对先验的假设可以总结为

$$
\pi(\bm{\lambda},\ \sigma^{2},\ \bm{\gamma},\ \bm{\omega}) = \pi(\bm{\lambda} \mid \sigma^{2},\ \bm{\gamma}) \pi(\sigma^{2}) \pi(\bm{\gamma} \mid \bm{\omega}) \pi(\bm{\omega})
$$

通过以上先验我们可以得到它们的后验：

$$
\begin{equation}
\bm{\lambda} \mid \text{data},\ \sigma^{2},\ \bm{\gamma},\ \bm{\omega} \sim \mathcal{N}\left(\underbrace{(\bm{C}^{\top} \bm{C} + \bm{D})^{-1} \bm{C}^{\top} \bm{\mu}_{\bm{R}}}_{\bm{\widehat{\lambda}}},\ \underbrace{\sigma^{2} (\bm{C}^{\top} \bm{C} + \bm{D})^{-1}}_{\widehat{\sigma}^{2}(\bm{\widehat{\lambda}})} \right) 
\end{equation}
$$

$$
\begin{equation}
\begin{split}
\frac{p(\gamma_j = 1 \mid \text{data},\ \bm{\lambda},\ \sigma^{2},\ \bm{\omega},\ \bm{\gamma}_{-j})}{p(\gamma_j = 0 \mid \text{data},\ \bm{\lambda},\ \sigma^{2},\ \bm{\omega},\ \bm{\gamma}_{-j})} = \frac{\omega_j}{1 - \omega_j} \frac{p(\lambda_j \mid \gamma_j = 1,\ \sigma^{2})}{p(\lambda_j \mid \gamma_j = 0,\ \sigma^{2})} := \xi_j \\
\implies p(\gamma_j = 1 \mid \text{data},\ \bm{\lambda},\ \sigma^{2},\ \bm{\omega},\ \bm{\gamma}_{-j}) = \frac{\xi_j}{1 + \xi_j}
\end{split} \label{12}
\end{equation}
$$

$$
\begin{equation}
\omega_j \mid \text{data},\ \bm{\lambda},\ \sigma^{2},\ \bm{\gamma} \sim \text{Beta}(\gamma_j + a_{\bm{\omega}},\ 1 - \gamma_j + b_{\bm{\omega}})
\end{equation}
$$

$$
\begin{equation}
\sigma^{2} \mid \text{data},\ \bm{\gamma},\ \bm{\omega} \sim \mathcal{IG}\left(\frac{N + K + 1}{2},\ \frac{\text{SSR}}{2} \right) 
\end{equation}
$$

其中

$$
\begin{equation}
\text{SSR} = (\bm{\mu}_{\bm{R}} - \bm{C} \bm{\lambda})^{\top} (\bm{\mu}_{\bm{R}} - \bm{C} \bm{\lambda}) + \bm{\lambda}^{\top} \bm{D} \bm{\lambda}
\end{equation}
$$

**当因子 $j$ 与 test assets 的相关性弱（弱因子），$\psi_j \to 0$，$\bm{D}$ 对应的对角线元素 $r(\gamma_j)\psi_j ^{-1}$ 趋于无穷，那么 $(\bm{C}^{\top} \bm{C} + \bm{D})^{-1}$ 中的对应元素就会趋于 $0$，这样我们得到的 $\lambda_j$ 的后验均值 $\widehat{\lambda}_{j}$ 就趋于 $0$。而对于比较强的因子，$\bm{D}$ 的存在也为风险价格的估计提供了 shrinkage。**


有了以上后验，我们同样可以用 Gibbs sampling 的方法按顺序对变量进行交替采样：

1. 为 $\sigma^{2},\ \bm{\omega}$ 设定初值；
2. 循环：依次对 $\bm{\Sigma}_{\bm{Y}},\ \bm{\mu}_{\bm{Y}},\ \bm{\lambda},\ \bm{\gamma},\ \bm{\omega},\ \sigma^{2}$ 采样。

> [!TIP|label:提示]
> 在设定 $a_{\bm{\omega}} = b_{\bm{\omega}} = 1$ 的情况下，$\bm{\omega}$ 的初值为 $\frac{1}{2} \cdot \bm{1}_{K}$，即每个因子一开始都有 $\frac{1}{2}$ 的概率被选到。
> 
> 对于弱因子 $j$，$\eqref{12}$ 式中 $p(\lambda_j \mid \gamma_j = 1,\ \sigma^{2})$ 与 $p(\lambda_j \mid \gamma_j = 0,\ \sigma^{2})$ 是比较接近的，因为不管我们包不包括它，我们抽样得到的风险价格都会集中在 $0$ 附近，因此 $\xi_j \approx \frac{\omega_j}{1 - \omega_j}$；而如果 $j$ 是个比较强的因子，我们抽样得到的风险价格应该是倾向于偏离 $0$ 的，这就导致 $p(\lambda_j \mid \gamma_j = 1,\ \sigma^{2}) \gg p(\lambda_j \mid \gamma_j = 0,\ \sigma^{2})$，即 $\xi_j \gg \frac{\omega_j}{1 - \omega_j}$。
> 
> 根据 $\eqref{12}$ 式，$\xi_j$ 越大，$\gamma_j = 1$ 的概率会越大，如果抽样得到的 $\gamma_j$ 一直是 $1$，那么在 $\omega_j$ 的后验分布中，参数 $a$ 会越来越大，参数 $b$ 则维持不变，这导致期望 $\frac{a}{a + b}$ 会越来越大，我们包括因子 $j$ 的期望概率也就越大；反之，如果 $\gamma_j$ 一直是 $0$，那么在 $\omega_j$ 的后验分布中，参数 $b$ 会越来越大，参数 $a$ 则维持不变，这导致期望 $\frac{a}{a + b}$ 会越来越小，我们包括因子 $j$ 的期望概率也就越小。
>
> 总的来说，**弱因子被包括进模型的概率会越来越小，强因子被包括进模型的概率会越来越大。**

至此，通过不断地循环，我们成功得到了 $\bm{\gamma}$ 的后验概率。

![](drawio/bayesian_solutions_for_the_factor_zoo4.png)

#### 超参数的含义

SDF 隐含的最大夏普比的平方为 SDF 的方差，即 $\bm{\lambda}_{\bm{f}}^{\top} \bm{\Sigma}_{\bm{f}} \bm{\lambda}_{\bm{f}}$，而定价误差 $\bm{\alpha}$ 的夏普比平方为 $\bm{\alpha}^{\top} \bm{\Sigma}_{\bm{R}}^{-1} \bm{\alpha}$，则在先验下我们有：

$$
\begin{aligned}
\frac{\E_{\pi}(\text{SR}_{\bm{f}}^{2} \mid \bm{\gamma},\ \sigma^{2})}{\E_{\pi}(\text{SR}_{\bm{\alpha}}^{2} \mid \sigma^{2})} &= \frac{\E_{\pi}(\bm{\lambda}_{\bm{f}}^{\top} \bm{\lambda}_{\bm{f}} \mid \bm{\gamma},\ \sigma^{2})}{\E_{\pi}(\bm{\alpha}^{\top} \bm{\Sigma}_{\bm{R}}^{-1} \bm{\alpha} \mid \sigma^{2})} \\
&= \frac{\sum_{k=1}^{K} r(\gamma_k) \psi_k \sigma^{2}}{\sum_{n=1}^{N} \sigma^{2}} \\
&= \frac{\psi \sum_{k=1}^{K} r(\gamma_k) \bm{\tilde{\rho}}_{k}^{\top}\bm{\tilde{\rho}}_{k}}{N} \\
\end{aligned}
$$

即超参数 $\psi$ 的大小代表了我们对夏普比的先验判断。

> [!TIP|label:提示]
> 由于对 test assets 和因子都做了标准化处理，且我们假设的先验是因子的风险价格是互相独立的，因此 $\bm{\Sigma}_{\bm{f}} = \bm{I}_{K}$。

> $\bm{\alpha}$ 夏普比平方的计算有待考证。

> test assets 的夏普比平方比上定价误差的夏普比平方代表了什么？

$\bm{\omega}$ 的先验 $\pi(\bm{\omega})$ 中，参数 $a_{\bm{\omega}}$ 与 $b_{\bm{\omega}}$ 代表了我们对模型 sparsity 的判断（见[上方](#引入-ω-辅助-γ-抽样)），同时也会对夏普比的先验判断有影响 （模型越 sparse，先验夏普比越小），因为 $\E_{\pi}(\text{SR}_{\bm{f}}^{2} \mid \sigma^{2}) = \frac{a_{\omega}}{a_{\omega} + b_{\omega}} \psi \sigma^{2} \sum_{k=1}^{K} \bm{\tilde{\rho}}_{k}^{\top}\bm{\tilde{\rho}}_{k}$。

### 模型集成

当我们抽样出不同的模型（不同的变量选择），我们可以用模型集成的方式综合考虑这些模型。当我们关心某个变量 $\Delta$ （可以是风险价格，风险溢价，或是最大夏普比等），根据贝叶斯定理我们有

$$
\begin{equation}
\E(\Delta \mid \text{data}) = \sum\limits_{m \in \mathcal{M}} \E(\Delta \mid \text{data},\ \text{model} = m) \operatorname{Pr}(\text{model} = m \mid \text{data}) \label{16}
\end{equation}
$$

其中 

$$
\begin{equation}
\E(\Delta \mid \text{data},\ \text{model} = m) = \lim\limits_{L \to \infty} \frac{1}{L} \sum\limits_{l=1}^{L} \Delta(\bm{\theta}_{l}^{(m)}) 
\end{equation}
$$

$\left\{\bm{\theta}_{l}^{(m)} \right\}_{l=1}^{L}$ 代表从模型 $m$ 的后验分布中的 $L$ 次参数抽样。

这样得到的估计被称为 **BMA（Bayesian Model Averaging）**，**实际上就是对所有抽样出来的模型简单取平均，只不过有的模型被抽的次数多，自然在 $\eqref{16}$ 式中权重高，这是一种减少过拟合的方法。**

> [!TIP|label:BMA 的好处]
> 1. 当我们在同一个风险上（比如在公司盈利能力的风险上）有多个因子 candidate 时，做 BMA 是一个很好的方法，它能通过加权平均的方式最大化 SDF 对截面定价的信噪比。
> 2. BMA 在平方误差函数下是最优的；
> 3. BMA 给出的分布与真实的 DGP（Data-Generating Process）之间的 KL 散度最小。

> 只知结论，不知为何。


## 实验

### 模拟

作者首先选取 Fama-French 的 5x5 组合作为 test assets，并假设只有 Fama 三因子中的 HML 是有用的因子，用正态分布来生成 test assets 的收益率和 HML，即 DGP 如下

$$
\begin{equation}
\begin{pmatrix}	\bm{R}_{t} \\ f_{t,\ \text{HML}} \\\end{pmatrix} \overset{\text{i.i.d.}}{\sim} \mathcal{N} \left(\begin{bmatrix}	\bm{\overline{\mu}}_{\bm{R}} \\	\overline{f}_{\text{HML}} \\\end{bmatrix},\ \begin{bmatrix}	\bm{\widehat{\Sigma}}_{\bm{R}} & \bm{\widehat{C}}_{\! \text{HML}} \\ \bm{\widehat{C}}_{\! \text{HML}}^{\top} & \widehat{\sigma}_{\text{HML}}^{2} \\\end{bmatrix} \right) 
\end{equation}
$$

其中正态分布内的参数都是根据真实数据计算的，但 DGP 生成的数据属于模拟数据（虚假的）。

对于真实的收益率均值，如果考虑模型是正确的，那么真实值就等于估计值；如果考虑模型是错误的，那么真实值用样本均值来代替，即

$$
\begin{equation}
\bm{\mu}_{\bm{R}} = 
\begin{cases}
    \widehat{\lambda}_{c} \bm{1}_{N} + \bm{\widehat{C}}_{\text{HML}} \widehat{\lambda}_{\text{HML}},\ &\text{if the model is correct} \\
    \bm{\overline{\mu}}_{\bm{R}},\ &\text{if the model is misspecified} \\
\end{cases} \label{19}
\end{equation}
$$

其次，作者还设计了一个弱因子：

$$
f_{t,\ \text{useless}} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,\ (1\%)^{2})
$$

作者使用 1963 年 7 月到 2017 年 12 月的月频数据来计算 DGP 的参数，但由于是模拟，样本理论上可以无限多，于是作者分别研究了 $T = 100,\ 200,\ 600,\ 1,000,\ 20,000$ 的情况。

> [!NOTE|label:注意]
> $T = 600$ 已经相当于 50 年的数据了，$T = 1,000$ 和 $T = 20,000$ 是真实数据所达不到的样本长度。

#### 比较频率派与贝叶斯派对弱因子的识别能力

作者首先利用真实数据估计出因子风险价格的“真实值” $\lambda^{*}$，即用真实数据对 $\eqref{19}$ 式中模型正确的情况分别进行 OLS 和 GLS 的估计；接着使用不同模型对模拟数据进行操作，得到因子风险价格的估计值，并进行不同置信水平的假设检验（$H_0: \lambda = \lambda^{*}$）。对于调整 R 方作者也进行了相同的操作，即先用真实数据得到“真实值”，再用模拟数据比较不同模型得出的调整 R 方。

对假设的 DGP 做了 2000 次模拟，作者得到下面的结果：

![](image/2023-04-05-19-19-14.png)

其中对于因子的风险价格我们用拒绝原假设的频率来比较，比如标橙的 $0.585$ 代表在 OLS 假设下，频率派的估计在 5\% 的置信水平上有 58.5\% 都拒绝了原假设 $H_0: \lambda_{\text{useless}} = \lambda_{\text{useless}}^{*} = 0$，即 2000 次拒绝了 1170 次，也就是 2000 次中有 1170 次都会认为弱因子好使。同样地，在 GLS 假设下，频率派估计有 86.5\% 的情况都会把弱因子错误地包括进去，这说明**频率派的估计基本无法识别弱因子；而不管是用均匀先验还是正态先验，贝叶斯派的估计在各个假设下都能很好地识别弱因子。**

对于调整 R 方我们主要看它的 5\% 置信区间是否够收敛于真实的调整 R 方。在 OLS（GLS）假设下，真实值为 $43.87\%$（$6.69\%$），可以看到**贝叶斯派得到的模型调整 R 方会比频率派更收敛到真实值。**

#### 比较不同先验下的变量选择能力

检验一个变量是否应该被纳入模型可以用**贝叶斯因子（Bayes factor）**：

$$
\text{BF}_{i} = p(\gamma_i = 1 \mid \text{data})
$$

**当贝叶斯因子超过某个阈值，我们认为应该包括这个变量。**

于是对于贝叶斯派估计，使用不同先验的变量选择结果如下：

![](image/2023-04-05-19-30-00.png)

最上方代表不同的阈值，标橙的这些数值代表变量被选择的频率。可以看到**使用钉板先验（正态先验）对于强因子的选择是比均匀先验要准确的，对于弱因子的选择，使用均匀先验和正态先验可以说是天差地别。**

### 实证

作者使用了 51 因子（34 个可交易，17 个不可交易）和 60 个 test assets（34 个可交易因子，26 个单变量排序得到的多空组合），数据集从 1973 年 10 月到 2016 年 12 月（$T \approx 600$）。实证中使用的是 GLS 假设下的贝叶斯估计。

#### 因子的后验概率与风险价格的后验均值

![](image/2023-04-06-11-00-00.png)

上图描述了 51 个因子被选择（$\gamma_j = 1$）的后验概率是如何随先验夏普比变化而变化的。其中 **BEH_PEAD 是盈余公告漂移（Postearnings Announcement Drift）因子，这个因子试图度量投资者的有限注意力**（Daniel 等，2020），后验概率遥遥领先，高于 70\%；排在第二位的 MKT 是市场因子；CMA_star 则是一个投资因子。

可以看到，少数几个因子后验概率随先验夏普比的增大而升高，当先验夏普比大到一定程度后后验概率同时开始降低；很多因子后验概率一直维持在 50\% 左右；还有很多因子后验概率一直随先验夏普比的增大而降低。这说明三个问题：

1. 在不怎么 shrinking（$\psi$ 比较大）的情况下，即使是有效的因子后验概率也会向 50\% 收敛（作者并没有探究，需要把 $\psi$ 再大一些的情况画出来才能确认）。我们要求高的夏普比，就需要更有效的因子，换句话说，**使用这些因子来得到高的夏普比并不现实**；
2. **很多因子最多是弱因子（weakly identified at best）**；
3. 还有**很多因子对于定价作者选择的 test assets 是无用的**（不是 SDF 的一部分）。

这些因子风险价格的具体后验概率和后验均值为

![](image/2023-04-06-16-28-09.png)

可以看到，<strong>对于那些弱因子，它们风险价格的后验均值基本为 $0$</strong>，只有当先验夏普比比较高时才会偏离 $0$，也就是说，**当我们要求高的夏普比时，对弱因子的估计就不再稳健。**

#### 对比传统模型

![](image/2023-04-06-19-29-44.png)

上图左半边为不同夏普先验下 BMA 的各项指标，右半边为传统模型的各项指标。Panel A 是样本内，Panel B 和 C 都是样本外（这里的样本外指的是截面的样本外，而非时序样本外）。标蓝和标红分别为指标最差和最好的模型。

可以看到**先验夏普比为 3.5 时，样本内的效果最好，比任意传统模型都要好，且在样本外的效果也不差**，Panel B 中也是比所有传统模型要好，Panel C 中除了 MAPE（Mean Absolute Pricing Error）外也是；**不同夏普先验下 BMA 的样本内和样本外效果都不错，即使最差的那些也和传统模型中最好的相差不多。**

#### 对比 KNS（2020）

*Shrinking the Cross-Section*（Kozak 等，2020，以下简称 KNS）也是通过一个相似的先验去对后验估计进行 shrinking，它们在时序上的样本外效果比传统模型强，因此本文直接与 KNS（2020）进行对比。

![](image/2023-04-06-21-42-18.png)

上图分别展示了 BMA 和 KNS（2020）在样本外的调整 R 方热力图，横坐标是先验夏普比，纵坐标是 KNS（2020）中主成分因子的数量，$\text{CV}_{3}$ 代表通过将全部数据分三组交叉验证得到的最优参数（见过样本外）。左图是新数据作为样本内，旧数据作为样本外；右图则是旧数据作为样本内，新数据作为样本外。

可以看到**在相同的先验夏普比下，BMA 比 KNS（2020）的交叉验证参数 $\text{CV}_{3}$ 要强，BMA 的 $\text{CV}_{3}$ 也比 KNS（2020）的要强；在不同的先验夏普比下，BMA 大多都比 KNS（2020）竖直方向上最好的情况要强，尤其是在我们使用新数据作为样本外的时候（右图）。**

> [!NOTE|label:注意]
> KNS（2020）和本文的一个区别在于 KNS（2020）直接对 test assets 做 PCA，是没有用到不可交易因子的，也许是这些因子的存在提高了本文模型的能力。

#### 选择 VS. 聚合

在频率派眼中，是存在一个最优模型的，但在计算 BMA 的时候究竟存不存在一个远超其他模型的模型呢？答案是不存在的。

![](image/2023-04-07-12-23-18.png)

上图展示了前 2,000 个模型的后验概率（被抽样的次数比上总抽样数），横轴是模型的排名（经过 log 伸缩处理），纵轴是后验概率。

可以看到**即使是最大的后验概率也只有 0.0011\%，且后验概率下降的速度非常慢**，前 9 个模型都是 0.0011\%，第 10 个到第 125 个是 0.0009\%，第 126 个到第 1687 个是 0.0007\%，之后是 0.0004\%。**因此没有一个 clear winner，对模型做聚合比选择一个最佳模型更适合。**

#### 因子数量和后验夏普比的分布

![](image/2023-04-07-12-36-26.png)

上图左边展示了不同夏普先验下因子数量的分布，右边展示了不同夏普先验下后验夏普比的分布。

**当先验夏普比在 1 到 3 之间时，因子数量的后验均值集中在 23 到 25，也就是说平均下来我们应该选择 24 个左右的因子**；而当先验夏普比到 3.5，因子数量的分布有一个非常大的左移，也就是说**当我们要求很高的夏普比，对因子风险溢价的先验就趋近于均匀先验了**（见 $\eqref{7}$ 式，先验夏普比高即 $\psi$ 高），而前面我们已经探讨过[均匀先验的坏处](#对风险价格使用均匀先验的坏处)，即**弱因子会被错误识别为强因子，这就导致强因子都被弱因子“挤”出去了。**

**对于最大夏普比，作者得到的后验均值跟先验差不太多，且分布都不会太分散，说明不会出现非常高的夏普比。**

![](image/2023-04-07-12-51-49.png)

上表展示了选择的因子中可交易因子和不可交易因子的数量，**平均可交易因子和不可交易因子各选一半。**

#### PCA 以及 RP-PCA 构建的隐因子是否有效

对 test assets 做 PCA，作者在原先 51 个因子的基础上加入了前 5 个隐因子和 2 个人工制造的无用因子（见[模拟](#模拟)）。

![](image/2023-04-07-13-25-38.png)

可以看到**这 5 个隐因子基本没什么用**，甚至不如人工制造的无用因子。

而将 PCA 换成 RP-PCA（Lettau 和 Pelger，2020），结果则大不相同：

![](image/2023-04-07-13-27-39.png)

尽管仍然有 3 个隐因子没什么用，<strong>有 2 个隐因子直接杀到了前面。</strong>同时，这 2 个隐因子并没有“挤”掉原先强势的那些因子，说明**这 2 个隐因子与原先强势的因子关系并不大。**

> 这 2 个隐因子对应的风险价格都是负的，这说明了什么？

## 可能的扩展

1. 如果想要限制最大夏普比，可以将钉板先验中的正态分布换成 Beta 分布，因为 Beta 分布横向有界（正态分布横向可以取到 $\pm \infty$）；
2. 本文对因子风险价格的先验是正态分布，可以扩展到一些厚尾的分布比如柯西分布；
3. 可以加入一些时变的参数，但这会让计算复杂度大大提高。

## 可能存在的问题

1. 由于有不可交易因子的存在，风险价格并不能和切点组合的权重联系起来，无法直接指导投资，而 KNS（2020）可以；
2. 因子和 test assets 收益率是同期的？
3. 均值 $\bm{\mu}$ 和协方差 $\bm{C}$ 都是抽样得到的，也就是说真实值是抽样得到的，这样计算 R 方是否有点流氓？

## 参考文献

Daniel, K., Hirshleifer, D., & Sun, L. (2020). Short- and Long-Horizon Behavioral Factors. The Review of Financial Studies, 33(4), 1673–1736. https://doi.org/10.1093/rfs/hhz069

Kozak, S., Nagel, S., & Santosh, S. (2020). Shrinking the cross-section. Journal of Financial Economics, 135(2), 271–292. https://doi.org/10.1016/j.jfineco.2019.06.008

Lettau, M., & Pelger, M. (2020). Factors That Fit the Time Series and Cross-Section of Stock Returns. The Review of Financial Studies, 33(5), 2274–2325. https://doi.org/10.1093/rfs/hhaa020
