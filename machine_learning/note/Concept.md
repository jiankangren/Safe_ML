### Shapley Value 

The Shapley value is one way to distribute the total gains to the players, assuming that they all collaborate.  其实就是**各种情况下的边际贡献**

* A  set  N (of n players/features?)

* characteristic  function $v$ that maps subsets of players to the real numbers, and $v(\empty)=0$.
  *  $v(S)$, called the worth of coalition describes the total expected sum of payoffs the members of $S$ can obtain by cooperation.

* According to the Shapley value, the amount that player i gets given in a coalitional game $(v,N)$ is    

  ​				$\phi_i(v)=\sum_{S\subseteq  N\setminus i}\left[v(S\cup i)-v(S)\right]\frac{|S|!(n-|S|-1)!}{n!}$


#### Example 

Assume 3 players,  with that 

* $v(1)=100, v(2)=125,v(3)=50$
* $v(12)=270,v(23)=350,v(13)=375$
* $v(1,2,3)=500$

we have,  $n!=6$.  For player 1

1. $S=\empty$,  表示1 是第一个加入，那么 可能情况是 123 或 132，  i.e.,  $\frac{|S|!(n-|S|-1)!}{n!}=\frac{(2)!}{3!}=1/3$  and  $v(S\cup i)-v(S)=100$.
2. $S=2$,  接下来是1，那么最后只有3一种可能i.e., $\frac{|S|!(n-|S|-1)!}{n!}=\frac{(1)!}{3!}=1/6$, $v(S\cup i)-v(S)=145$
3. $S=3$,  接下来是1，那么最后只有2一种可能i.e., $\frac{|S|!(n-|S|-1)!}{n!}=\frac{(1)!}{3!}=1/6$, $v(S\cup i)-v(S)=325$
4. $S=23$,  接下来是1，那么最后只有$\empty$一种可能i.e., $\frac{|S|!(n-|S|-1)!}{n!}=\frac{(1)!}{3!}=1/3$, $v(S\cup i)-v(S)=500-350=150$

Finally, we have 

​	$\phi_1(v)=100/3+145/6+325/6+150/3=970/6$

