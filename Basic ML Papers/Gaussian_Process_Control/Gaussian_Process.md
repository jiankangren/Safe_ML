##高斯过程（Gaussian process）-表示函数的分布

https://zhuanlan.zhihu.com/p/32152162
* 机器学习的常见做法是把函数参数化，然后用产生的参数建模来规避分布表（如线性回归的权重）

* Gaussian process 直接对函数建模生成非参数模型-产生的一个突出优势就是它不仅能模拟任何黑盒函数，还能模拟不确定性 i.e., 预测的confidence。

1. 假设有一个隐藏函数：$$f:\mathbb R\rightarrow \mathbb R$$，我们要对它建模；
2. $$x=[x_1,…,x_N]^T，y=[y_1,…,y_N]^T$$，其中$$y_i=f(x_i)$$
3. 我们要计算函数f在某些未知输入$$x_∗$$上的值。


### 用高斯建模函数：
输入空间中的每个点都与一个随机变量相关联，而它们的联合分布可以被作为多元高斯分布建模
$$
\left(\begin{array}
~y_0\\
y_1\\
\ldots\\
y_N
\end{array}\right)\sim\mathcal N
\left(\left(\begin{array}
~0\\
0\\
\ldots\\
0
\end{array}\right), \left(\begin{array}
~k(x_1,x_1)& k(x_1,x_2)&\ldots&k(x_1,x_N)\\
k(x_2,x_1)&k(x_2,x_2)&\ldots\\
\ldots\\
k(x_N,x_1)& k(x_N,x_2)&\ldots&k(x_N,x_N)
\end{array}\right) \right)
$$

对于确定的点 $$y_1$$ 它是分布的一个采样结果。

#### 核函数
* 平方形式的核函数（最简形式）
$$
k(x,x')=\exp(\frac{-(x-x')^2}{2})
$$


### 训练数据来模拟那个隐藏函数，从而预测y值
因为样本有限$$(\pmb x,\pmb y )$$，预测集合 $$(\pmb x_*,\pmb y_* )$$, 联合分布：
$$
\left(\begin{array}
~\pmb y\\
\pmb y_*
\end{array}\right)\sim\mathcal N \left(\left(\begin{array}
~m(\pmb x)\\
m(\pmb x_*)\\
\end{array}\right), \left(\begin{array}
~\pmb K& \pmb K_*\\
\pmb K_*^T& \pmb K_{**}
\end{array}\right) \right)\\
$$
#### mean function $$m(x)=0$$ i.e., 给定x 对应的y的均值

$$
\pmb K=k(x,x)，\pmb K_∗=k(x,x_∗)， \pmb K_{∗∗}=k(x_∗,x_∗)\\
\Rightarrow \mbox{现在，模型成了} p(\pmb y,\pmb y_∗|\pmb x,\pmb x_∗) \mbox{是一个关于x x_*的函数,但是y其实不用求啊}\\
\Rightarrow p(\pmb y_∗|\pmb x,\pmb y,\pmb x_∗)\sim \mathcal N(\pmb y_*|u_*,\Sigma_*)\\
u_*=m(\pmb x_*)+\pmb K^T\pmb K^{-1}(y-m(\pmb x))\\
\Sigma_*=\pmb K_{**}-\pmb K^T_*\pmb K^{-1} \pmb K\\
这就是基于先验分布和观察值计算出的关于y∗的后验分布
$$

每个 y的confidence 边界可以是 $$\bar y+/-k\times \sigma_y$$