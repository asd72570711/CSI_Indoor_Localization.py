# XGBoost
***
### 基本模型與參數

XGBoost = e**X**treme **G**radient **Boost**ing

用於**監督式學習** (Supervised Learning) 問題

利用多個特徵 (Features) 之訓練資料 (Training Data) 來預測目標變數

<br />

基本之監督式學習數學架構如下表示：

<br />

<img src="https://latex.codecogs.com/gif.latex?\large&space;\hat&space;y_i&space;=&space;\sum_j\theta_jx_{ij}" title="\large \hat y_i = \sum_j\theta_jx_{ij}" />

<br />

其中 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;x_i" title="\large x_i" /> 為訓練資料，<img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;y_i" title="\large y_i" /> 為標籤 (Label)

&theta; 是要利用資料來學習的參數，用以擬合 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;x_i" title="\large x_i" /> 和 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;y_i" title="\large y_i" /> 

目的即是找出最佳的 &theta;

因此我們需要定義一個**目標方程** (Objectiive Function) 來測量模型 (Model) 與訓練集的擬合效果

<br />

目標方程通常是以兩項方程式組成，分別為**訓練損失函數** (Training Loss Function) 和**正規項** (Regularization Term)：

<br />

<img src="https://latex.codecogs.com/gif.latex?\large&space;obj(\theta)&space;=&space;L(\theta)&space;&plus;&space;\Omega(\theta)" title="\large obj(\theta) = L(\theta) + \Omega(\theta)" />

<br />

其中 L 為訓練損失函數，用以表示模型如何擬合訓練集

通常訓練損失函數選擇以 **MSE** (Mean Square Error) 表現，寫成：

<br />

<img src="https://latex.codecogs.com/svg.latex?\large&space;L(\theta)&space;=&space;\sum_i&space;(y_i-\hat&space;y_i)^2" title="\large L(\theta) = \sum_i (y_i-\hat y_i)^2" />

<br />

或者以**邏輯回歸** (Logistic Regression) 的方式表示：

<br />

<img src="https://latex.codecogs.com/svg.latex?\large&space;L(\theta)&space;=&space;\sum_i&space;[y_iln(1&plus;e^{-\hat&space;y_i})&plus;(1-y_i)ln(1&plus;e^{\hat&space;y_i})]" title="\large L(\theta) = \sum_i [y_iln(1+e^{-\hat y_i})+(1-y_i)ln(1+e^{\hat y_i})]" />

<br />

<br />

&Omega; 為正規項，是用來描述模型的**複雜性**，幫助避免**過度擬合** (Overfitting) 的情況發生

下方圖示進行簡單說明 L 和 &Omega;

![alt tag](https://i.imgur.com/AsV0DAI.png)

***
### 決策樹

#### CART

CART = **C**lassification **a**nd **R**egression **T**rees

![Imgur](https://i.imgur.com/1P6pgEv.png)

如上圖所示，我們希望將五個家族成員進行分類，而分進不同的分類，則稱此分類為不同的**葉子 (Leaf)**

而每個葉子都有其權重，分配不同的權重給不同的葉子

決策樹 (Decision Trees) 主要有兩種類型

* 分類樹 (Classification Trees)
* 回歸樹 (Regression Trees)

**分類樹**的輸出是樣本的類標

**回歸樹**的輸出是一個實數

而 CART 則是上述兩種樹的結合

常理來說，一棵樹並不足以用來訓練，因此我們需要很多不同的樹

最直觀的方法即是將多顆樹之預測值 (Prediction Trees) 加總

形成模型的集合體 (Esemble Model)

![Imgur](https://i.imgur.com/q0N5yS2.png)

上圖是兩顆樹的集合，可以看到兩棵樹是可以重合加總的

以數學方程式表示之，可以將我們的模型寫作：

<img src="https://latex.codecogs.com/svg.latex?\large&space;\hat&space;y_i&space;=&space;\sum_{k=1}^K&space;f_k(x_i)&space;,&space;f_k\in&space;F" title="\large \hat y_i = \sum_{k=1}^K f_k(x_i) , f_k\in F" />

其中 K 是樹的數量，F 是所有可能的 CART 之集合，f 則是在集合 F 中的方程式

因此我們可以得到目標方程式：

<img src="https://latex.codecogs.com/svg.latex?\large&space;obj(\theta)&space;=&space;\sum^n_i&space;l(y_i,\hat&space;y_i)&plus;\sum^K_{k=1}\Omega(f_k)" title="\large obj(\theta) = \sum^n_i l(y_i,\hat y_i)+\sum^K_{k=1}\Omega(f_k)" />


此模型與**隨機森林** **(Random Forests)** 相同，差別在於訓練方式的不同

***
### Tree Boosting

訓練樹的方法 = 訓練監督式學習 = **定義且最佳化目標方程**

令目標方程如下：

<img src="https://latex.codecogs.com/svg.latex?\large&space;obj&space;=&space;\sum^n_{i=1}l(y_i,\hat&space;y_i^{(t)})&plus;\sum&space;^t_{i=1}\Omega(f_i)" title="\large obj = \sum^n_{i=1}l(y_i,\hat y_i^{(t)})+\sum ^t_{i=1}\Omega(f_i)" />

其中我們必須學習方程式 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;f_i" title="\large f_i" />

每個 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;f_i" title="\large f_i" /> 包含了樹的結構以及葉子的分數

相較於傳統利用梯度法最佳化的問題，樹結構的學習顯得更為艱難

同時訓練所有樹也非常困難

取而代之，固定我們已經學習的樹，並同時加進新的樹，是一個更是當的策略

我們將第 t 步的預測值寫作 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\hat&space;y_i^{(t)}" title="\large \hat y_i^{(t)}" /> 

<img src="https://latex.codecogs.com/svg.latex?\large&space;\hat&space;y_i^{(0)}&space;=&space;0" title="\large \hat y_i^{(0)} = 0" />

<img src="https://latex.codecogs.com/svg.latex?\large&space;\hat&space;y_i^{(1)}&space;=&space;f_1(x_i)&space;=&space;\hat&space;y_i^{(0)}&space;&plus;&space;f_1(x_i)" title="\large \hat y_i^{(1)} = f_1(x_i) = \hat y_i^{(0)} + f_1(x_i)" />

<img src="https://latex.codecogs.com/svg.latex?\large&space;\hat&space;y_i^{(2)}&space;=&space;f_1(x_i)&space;&plus;&space;f_2(x_i)&space;=&space;\hat&space;y_i^{(1)}&space;&plus;&space;f_2(x_i)" title="\large \hat y_i^{(2)} = f_1(x_i) + f_2(x_i) = \hat y_i^{(1)} + f_2(x_i)" />

<img src="https://latex.codecogs.com/svg.latex?\large&space;\vdots" title="\large \vdots" />

<img src="https://latex.codecogs.com/svg.latex?\large&space;\hat&space;y_i^{(t)}&space;=&space;\sum^t_{k=1}f_k(x_i)&space;=&space;\hat&space;y_i^{(t-1)}&space;&plus;&space;f_t(x_i)" title="\large \hat y_i^{(t)} = \sum^t_{k=1}f_k(x_i) = \hat y_i^{(t-1)} + f_t(x_i)" />

每一個步驟加上能將我們的目標方程最佳化的樹，因此目標方程寫作：

<img src="https://latex.codecogs.com/svg.latex?\large&space;\begin{aligned}&space;obj^{(t)}&space;&=&space;\sum^n_{i=1}l(y_i,\hat&space;y_i^{(t)})&space;&plus;&space;\sum^t_{i=1}\Omega(f_i)&space;\\&space;&=&space;\sum^n_{i=1}l(y_i,\hat&space;y_i^{(t-1)}&space;&plus;&space;f_t(x_i))&space;&plus;&space;\Omega(f_t)&space;&plus;&space;constant&space;\end{aligned}" title="\large \begin{aligned} obj^{(t)} &= \sum^n_{i=1}l(y_i,\hat y_i^{(t)}) + \sum^t_{i=1}\Omega(f_i) \\ &= \sum^n_{i=1}l(y_i,\hat y_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) + constant \end{aligned}" />

若我們將**均方誤差 (MSE)** 當作我們的損失函數，則目標方程變成：

<img src="https://latex.codecogs.com/svg.latex?\large&space;\begin{aligned}&space;obj^{(t)}&space;&=&space;\sum^n_{i=1}(y_i-(\hat&space;y_i^{(t-1)}&plus;f_t(x_i)))^2&space;&plus;&space;\sum^t_{i=1}\Omega(f_i)&space;\\&space;&=&space;\sum^n_{i=1}[2(\hat&space;y_i^{(t-1)}-y_i)f_t(x_i)&plus;f_t(x_i)^2]&space;&plus;&space;\Omega(f_t)&space;&plus;&space;constant&space;\end{aligned}" title="\large \begin{aligned} obj^{(t)} &= \sum^n_{i=1}(y_i-(\hat y_i^{(t-1)}+f_t(x_i)))^2 + \sum^t_{i=1}\Omega(f_i) \\ &= \sum^n_{i=1}[2(\hat y_i^{(t-1)}-y_i)f_t(x_i)+f_t(x_i)^2] + \Omega(f_t) + constant \end{aligned}" />

此步驟即是將上方的損失函數 <img src="https://latex.codecogs.com/svg.latex?\inline&space;L(\theta)" title="L(\theta)" /> 代入上面的 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;obj^{(t)}" title="\large obj^{(t)}" /> 的通式

即

<img src="https://latex.codecogs.com/svg.latex?\large&space;L(\theta)&space;=&space;\sum_i(y_i-\hat&space;y_i)^2" title="\large L(\theta) = \sum_i(y_i-\hat y_i)^2" />

代入 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;obj^{(t)}" title="\large obj^{(t)}" /> 得到：

<img src="https://latex.codecogs.com/svg.latex?\large&space;\begin{aligned}&space;obj^{(t)}&space;=&&space;\sum_{i=1}^n(y_i-\hat&space;y_i)^2&space;&plus;&space;\sum_{i=1}^t\Omega(f_i)&space;\\&space;=&&space;\sum_{i=1}^n(y_i-(\hat&space;y^{(t-1)}_i&plus;f_t(x_i)))^2&space;&plus;&space;\sum_{i=1}^t\Omega(f_i)&space;\\&space;=&&space;\sum_{i=1}^n(y_i-(\hat&space;y^{(t-1)}_i&plus;f_t(x_i)))^2&space;&plus;&space;\Omega(f_t)&space;&plus;&space;constant&space;\\&space;=&&space;\sum_{i=1}^n(y_i-\hat&space;y^{(t-1)}_i)^2&space;&plus;&space;2(\hat&space;y_i^{(t-1)}-y_i)f_t(x_i)&space;&plus;&space;f_t(x_i)^2&space;&plus;\Omega(f_t)&space;&plus;&space;constant&space;\\&space;=&&space;\sum_{i=1}^n([2(\hat&space;y_i^{(t-1)}-y_i)f_t(x_i)&space;&plus;&space;f_t(x_i)^2]&space;&plus;\Omega(f_t)&space;&plus;&space;constant'&space;\\&space;\end{aligned}" title="\large \begin{aligned} obj^{(t)} =& \sum_{i=1}^n(y_i-\hat y_i)^2 + \sum_{i=1}^t\Omega(f_i) \\ =& \sum_{i=1}^n(y_i-(\hat y^{(t-1)}_i+f_t(x_i)))^2 + \sum_{i=1}^t\Omega(f_i) \\ =& \sum_{i=1}^n(y_i-(\hat y^{(t-1)}_i+f_t(x_i)))^2 + \Omega(f_t) + constant \\ =& \sum_{i=1}^n(y_i-\hat y^{(t-1)}_i)^2 + 2(\hat y_i^{(t-1)}-y_i)f_t(x_i) + f_t(x_i)^2 +\Omega(f_t) + constant \\ =& \sum_{i=1}^n([2(\hat y_i^{(t-1)}-y_i)f_t(x_i) + f_t(x_i)^2] +\Omega(f_t) + constant' \\ \end{aligned}" />

若將**邏輯回歸**當作損失函數，則目標方程顯得十分複雜

因此先將上方 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;obj^{(t)}" title="\large obj^{(t)}" /> 用**泰勒級數展開**至二階可得到：

<img src="https://latex.codecogs.com/svg.latex?\large&space;\begin{aligned}&space;obj^{(t)}&space;=&&space;\sum_{i=1}^nl(y_i,\hat&space;y_i^{(t)})&space;&plus;&space;\sum_{i=1}^t\Omega(f_i)&space;\\&space;=&&space;\sum_{i=1}^n[l(y_i,\hat&space;y_i^{(t-1)})&space;&plus;&space;\frac{\partial&space;l(y_i,\hat&space;y_i^{(t-1)})}{\partial(\hat&space;y_i^{(t-1)})}&space;f_t(x_i)&space;&plus;&space;\frac{1}{2}\frac{\partial&space;^2&space;l(y_i,\hat&space;y_i^{(t-1)})}{\partial&space;^2(\hat&space;y_i^{(t-1)})}&space;f_t(x_i)^2&space;&plus;&space;...]&space;&plus;&space;\Omega(f_t)&space;&plus;constant&space;\\&space;=&&space;\sum_{i=1}^n[l(y_i,\hat&space;y_i^{(t-1)})&space;&plus;&space;g_i&space;f_t(x_i)&space;&plus;&space;\frac{1}{2}h_i&space;f_t(x_i)^2&space;&plus;&space;...]&space;&plus;&space;\Omega(f_t)&space;&plus;constant&space;\\&space;=&&space;\sum_{i=1}^n[g_i&space;f_t(x_i)&space;&plus;&space;\frac{1}{2}h_i&space;f_t(x_i)^2&space;&plus;&space;...]&space;&plus;&space;\Omega(f_t)&space;&plus;constant'&space;\end{aligned}" title="\large \begin{aligned} obj^{(t)} =& \sum_{i=1}^nl(y_i,\hat y_i^{(t)}) + \sum_{i=1}^t\Omega(f_i) \\ =& \sum_{i=1}^n[l(y_i,\hat y_i^{(t-1)}) + \frac{\partial l(y_i,\hat y_i^{(t-1)})}{\partial(\hat y_i^{(t-1)})} f_t(x_i) + \frac{1}{2}\frac{\partial ^2 l(y_i,\hat y_i^{(t-1)})}{\partial ^2(\hat y_i^{(t-1)})} f_t(x_i)^2 + ...] + \Omega(f_t) +constant \\ =& \sum_{i=1}^n[l(y_i,\hat y_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2}h_i f_t(x_i)^2 + ...] + \Omega(f_t) +constant \\ =& \sum_{i=1}^n[g_i f_t(x_i) + \frac{1}{2}h_i f_t(x_i)^2 + ...] + \Omega(f_t) +constant' \end{aligned}" />
 
利用此表示法的優點在於最佳目標方程的值只與 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;g_i" title="\large g_i" /> 和 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;h_i" title="\large g_i" /> 有關

應用這種方法，我們可以最佳化每一種損失函數！

#### 模型複雜性

前面有提到訓練階段還有一個重要的部分，**正規項**

因此我們需要定義樹的複雜度 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;\Omega&space;(f)" title="\large \Omega (f)" /> 

為定義之，我們需要重新定義樹的定義 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;f(x)" title="\large f(x)" /> ：

<img src="https://latex.codecogs.com/svg.latex?\large&space;f_t(x)&space;=&space;w_{q(x)}&space;,&space;w&space;\in&space;R^T&space;,&space;q:R^d\rightarrow&space;{1,2,...,T}" title="\large f_t(x) = w_{q(x)} , w \in R^T , q:R^d\rightarrow {1,2,...,T}" />

<img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;w" title="\large w" /> 係葉子的分數所構成之向量，
<img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;q" title="\large q" /> 是每個資料對應到的葉子的方程式，
<img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;T" title="\large T" /> 為葉子的數目

在 XGBoost 的模型中，我們定義複雜度為：

<img src="https://latex.codecogs.com/svg.latex?\large&space;\Omega&space;(f)&space;=&space;\gamma&space;T&space;&plus;&space;\frac{1}{2}&space;\lambda&space;\sum^T_{j=1}w_j^2" title="\large \Omega (f) = \gamma T + \frac{1}{2} \lambda \sum^T_{j=1}w_j^2" />

當然還有其他對於複雜度的定義，但此時我們以這定義為例

#### 分數架構

透過上述那些方程式，我們可以重新寫出在第 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;t" title="\large t" /> 顆樹時的目標方程值為：

<img src="https://latex.codecogs.com/svg.latex?\large&space;obj^{(t)}&space;=&space;\sum^n_{i=1}[l(y_i,\hat&space;y_i^{(t-1)})&space;&plus;&space;g_if_t(x_i)&plus;\frac{1}{2}&space;h_if_t^2(x_i)]&plus;\gamma&space;T&space;&plus;\frac{1}{2}&space;\lambda&space;\sum^T_{j=1}w^2_j&space;&plus;&space;constant" title="\large obj^{(t)} = \sum^n_{i=1}[l(y_i,\hat y_i^{(t-1)}) + g_if_t(x_i)+\frac{1}{2} h_if_t^2(x_i)]+\gamma T +\frac{1}{2} \lambda \sum^T_{j=1}w^2_j + constant" />

將我們重定義的 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;f_t(x)" title="\large f_t(x)" /> 代入：

<img src="https://latex.codecogs.com/svg.latex?\large&space;\begin{aligned}&space;obj^{(t)}&space;=&&space;\sum^n_{i=1}[l(y_i,\hat&space;y_i^{(t-1)})&space;&plus;&space;g_iw_{q(x_i)}&plus;\frac{1}{2}&space;h_iw_{q(x_i)}^2]&plus;\gamma&space;T&space;&plus;\frac{1}{2}&space;\lambda&space;\sum^T_{j=1}w^2_j&space;&plus;&space;constant&space;\\&space;\approx&space;&&space;\sum^n_{i=1}[g_iw_{q(x_i)}&space;&plus;&space;\frac{1}{2}&space;h_iw_{q(x_i)}^2]&space;&plus;&space;\gamma&space;T&space;&plus;&space;\frac{1}{2}\lambda\sum^T_{j=1}w_j^2&space;\\&space;=&&space;\sum^T_{j=1}[(\sum_{i&space;\in&space;I_j}g_i)w_j&space;&plus;&space;\frac{1}{2}(\sum_{i&space;\in&space;I_j}h_i&plus;\lambda)w_j^2]&plus;\gamma&space;T&space;\\&space;\end{aligned}" title="\large \begin{aligned} obj^{(t)} =& \sum^n_{i=1}[l(y_i,\hat y_i^{(t-1)}) + g_iw_{q(x_i)}+\frac{1}{2} h_iw_{q(x_i)}^2]+\gamma T +\frac{1}{2} \lambda \sum^T_{j=1}w^2_j + constant \\ \approx & \sum^n_{i=1}[g_iw_{q(x_i)} + \frac{1}{2} h_iw_{q(x_i)}^2] + \gamma T + \frac{1}{2}\lambda\sum^T_{j=1}w_j^2 \\ =& \sum^T_{j=1}[(\sum_{i \in I_j}g_i)w_j + \frac{1}{2}(\sum_{i \in I_j}h_i+\lambda)w_j^2]+\gamma T \\ \end{aligned}" />

其中 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;I_j&space;=&space;\{i|q(x_i)&space;=&space;j\}" title="\large I_j = \{i|q(x_i) = j\}" />
是指派到第 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;j" title="\large j" /> 片葉子的指標資料點的集合

而方程式能夠合併是因為所有資料分配到某片葉子的得分都會相同，簡單來說即 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;w_{q(x_j)}&space;=&space;w_j" title="\large w_{q(x_j)} = w_j" />

令 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;G_j&space;=&space;\sum_{i&space;\in&space;I_j}g_i&space;,&space;H_j&space;=&space;\sum_{i&space;\in&space;I_j}h_i" title="\large G_j = \sum_{i \in I_j}g_i , H_j = \sum_{i \in I_j}h_i" /> 可將方程改寫成：

<img src="https://latex.codecogs.com/svg.latex?\large&space;obj^{(t)}&space;=&space;\sum^T_{j=1}[G_jw_j&plus;\frac{1}{2}(H_j&plus;\lambda)w_j^2]&space;&plus;&space;\gamma&space;T" title="\large obj^{(t)} = \sum^T_{j=1}[G_jw_j+\frac{1}{2}(H_j+\lambda)w_j^2] + \gamma T" />

式子中，<img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;w_j" title="\large w_j" /> 彼此獨立

而 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;G_jw_j&space;&plus;&space;\frac{1}{2}(H_j&plus;\lambda)w_j^2" title="\large G_jw_j + \frac{1}{2}(H_j+\lambda)w_j^2" /> 是 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;w_j" title="\large w_j" /> 的二次方程式

通過簡單的運算可以得出 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;w_j" title="\large w_j" /> 的最佳解：

<img src="https://latex.codecogs.com/svg.latex?\large&space;\begin{aligned}&space;G_jw_j&space;&plus;&space;\frac{1}{2}&space;(H_j&plus;\lambda)w_j^2&space;=&&space;0&space;\\&space;w_j[G_j&space;&plus;&space;\frac{1}{2}(H_j&plus;\lambda)w_j]&space;=&&space;0&space;\\&space;w_j(w_j&space;&plus;&space;\frac{2G_j}{H_j&plus;\lambda})&space;=&&space;0&space;\\&space;\end{aligned}" title="\large \begin{aligned} G_jw_j + \frac{1}{2} (H_j+\lambda)w_j^2 =& 0 \\ w_j[G_j + \frac{1}{2}(H_j+\lambda)w_j] =& 0 \\ w_j(w_j + \frac{2G_j}{H_j+\lambda}) =& 0 \\ \end{aligned}" />

<img src="https://latex.codecogs.com/svg.latex?\large&space;w_j&space;=&space;0&space;\&space;,&space;-\frac{2G_j}{H_j&plus;\lambda}" title="\large w_j = 0 \ , -\frac{2G_j}{H_j+\lambda}" />

我們將 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;w_j" title="\large w_j" /> 進行歸約，得到：

<img src="https://latex.codecogs.com/svg.latex?\large&space;w_j^*&space;=&space;-\frac{G_j}{H_j&plus;\lambda}&space;\\&space;obj^*&space;=&space;-\frac{1}{2}\sum^T_{j=1}\frac{G_j^2}{H_j&plus;\lambda}&space;&plus;&space;\gamma&space;T" title="\large w_j^* = -\frac{G_j}{H_j+\lambda} \\ obj^* = -\frac{1}{2}\sum^T_{j=1}\frac{G_j^2}{H_j+\lambda} + \gamma T" />

歸約(Reduction)：簡單來說就是將舊的問題換成新的問題，而解答一致

<img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;obj^*" title="\large obj^*" /> 判斷樹的架構 <img src="https://latex.codecogs.com/svg.latex?\inline&space;\large&space;q(x)" title="\large q(x)" /> 的好壞






