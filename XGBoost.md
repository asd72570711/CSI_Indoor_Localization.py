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

其中 x<sub>i</sub> 為訓練資料，y<sub>i</sub> 為標籤 (Label)

&theta; 是要利用資料來學習的參數，用以擬合 x<sub>i</sub> 和 y<sub>i</sub> 

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

此模型與**隨機森林(Random Forests)**相同，差別在於訓練方式的不同

***
### Tree Boosting
