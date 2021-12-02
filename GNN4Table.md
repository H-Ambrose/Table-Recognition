# GNN for Table Detection
## 2017 Table Recognition in Heterogeneous Documents using Machine Learning
**流程**：Optical character recognition (OCR) → Feature extraction → Neural networks training（AutoMLP）→ Post processing → Evaluation（ → Table structure analysis）  
**特征**：词级上下文特征的空白分布来识别文档图像中的表元素和非表元素（word level contextual features in terms of white space distribution）。主要包括单词与它的左、右、上、下邻居词元素之间的距离；单词的宽度和高度（这些特征的选择是基于观察到通常表元素之间的空白分布不同于非表元素之间的空白分布）。的像素距离。系统使用该信息将单词列表中的单个单词标记为表元素和非表元素。这些特征被保存在一个对象数组中，该数组具有以下值来构建特征向量。直接使用文本块的坐标信息来计算每个文本块的特征向量  
后处理：AutoMLP输出的分类结果准确率仅为64.2%，增加后处理，最终达到了95.08%的分类准确率。将当前单词的类标签修改为相邻区域内多数计数的类标签。e.g. 如果单词的左右邻居都被认为是“table”，那么考虑中的单词的标签也被更改为“table”。  

## DAS 2018 Comparing machine learning approaches for table recognition in historical register books
Clinchant et al.  
graph Conditional Random Fields (gCRF) vs GCN  

## DAS 2018 Table recognition in spreadsheets via a graph representation
Koci et al.  
remove and conquer algorithm  

## IWRR 2018 An invoice reading system using a graph convolutional network
Lohani et al.  
GCN  

## 2019 Table Detection in Invoice Documents by Graph Neural Networks
**输入与输出**：  
输入 – 文档实体的可见图(visibility graph)。G = (V, A)，V是节点的embedding，A是邻接矩阵。  
输出 – 文档实体的标签与邻接矩阵。其中每个元素为对应相邻点属于同一文档实体的概率。邻接矩阵是边分类结果：distinguish edges connecting nodes inside the same region, labelled with a 1 in the ground-truth against these edges that connect two different regions, labelled as 0.  

**特征**：  
G = (V, E)为可见性图。节点V集对应于检测到的文档实体。每个检测到的实体对应于一个7维向量，其中包含边界框位置及其内容的概率直方图(数字、字母或符号)，然后进行embedding。  
边集E表示节点之间的可见性关系。当且仅当边界框垂直或水平可见时(可以追踪一条水平或垂直的直线，而不与任何其他实体相交)，两个实体用一条边连接。使用这两个方向来检查可见性就足够了，因为它遵循文档中表格的组织方式。最后，覆盖超过四分之一页面高度的长边将被丢弃。  

网络结构：Graph Residual，Adjacency Layer，参考：Few-shot learning with graph neural networks  
Graph Residual中的邻接矩阵不更新，是利用邻接矩阵更新节点的embedding。最后加入一次Adjacency Learning更新一下邻接矩阵（邻接矩阵有两个(3个？)，2 hop和5 hop）  


# GNN for Table Recognition
ref: 参考了[这里](https://github.com/bljessica/paper-reading-records/blob/62e694df62e2217daf9d53cb358e97a7cc4c8e46/GNN4Table/notes.md)

## 《Rethinking Table Recognition using Graph Neural Networks》
### 介绍
结构分析是文档处理最重要的方面之一，它包含物理和逻辑布局分析，也包含解析或识别包括表格、菜谱和表单在内的复杂结构化布局  
在本文中，作者用图神经网络来处理这个问题  
#### 优点
+ 更具有普适性，因为此方法没有很多关于结构的强假设并且很接近人类的解释表格
+ 允许我们继续用图神经网络开发
#### 主要贡献
+ 规划表格识别问题为可以适用图神经网络的图问题
+ 设计一个新奇的可以综合图特征提取的CNN和高效顶点交互的GNN的架构
+ 介绍一个新奇的基于Monte Carlo的技术来减少训练中的记忆需求
+ 介绍一个综合数据集来弥补大规模数据集的缺失
+ 在两种先进水平的基于图的方法上进行测试，实验表明他们比基准网络表现得更好
### 数据集

## 《Complicated Table Structure Recognition》
### 背景
表格结构识别的目的是识别表格内部结构，这是让机器理解表格的关键步骤
### 问题
现有方法很难精准识别出PDF文件中的复杂表格。复杂表格包含占据至少两行或两列的表格单元。  
为了设法解决这个问题，作者提出了提出了一个新奇的用于识别PDF文件中的表格结构的图神经网络，命名为GraphTSR。  
具体地说，它将表格单元作为输入，然后通过预测表格单元之间的关系来识别表格结构。  
而且，为了更好地评估这个任务，作者从科学论文中构建了一个大规模表格结构识别数据集，命名为SciTSR，它包含来自PDF文件和和他们的相关结构标签的15000个表格。  
广泛的实验表明作者提出的模型对于复杂表格是很高效的，并且在基准数据集和作者的新数据集上比先进水平表现的更好。
### 介绍
识别出的机器可以理解的表格有许多可能的应用，包括回答问题，对话系统和表格转文本。  
现有的PDF格式表格识别方法可以分为两类：基于规则的方法和数据驱动的方法。然而，现有方法很难精确识别PDF文件中的复杂表格结构。  
包含跨表格单元的表格就叫做复杂表格。虽然跨表格单元在复杂表格中通常占比很小，但是他们比普通单元格包含更重要的语义信息，因为他们更有可能是表格中的表头。表头对于理解表格非常重要。因此，识别复杂表格结构是一个更重要的要解决的问题。  
为了解决上述问题，作者提出了一个新奇的图神经网络模型，它可以将此任务再表示成一个图上的边预测问题形式。具体来说，它将一个表格通过一堆图attention块编码，然后通过预测表格单元之间的联系来识别表格结构。另外，因为这个任务存在不可获取的训练数据，作者构建了一个新的大规模数据集SciTSR。  
作者的主要贡献：
+ 提出了一个新奇的图神经网络模型来识别PDF文件中的表格结构，尤其是复杂表格。广泛的实验表明作者提出的模型比先进水平表现的更好
+ 从科学论文中构建了一个大规模表格结构识别数据集，命名为SciTSR，它包含来自PDF文件和和他们的相关结构标签的15000个表格。在作者的认知中，这是第一个用于PDF文件中的表格结构识别的大规模数据集
### 相关工作
#### 现有方法
现有的PDF格式表格识别方法可以分为两类：基于规则的方法和数据驱动的方法。
+ 基于规则的方法
  + 利用规则行和文本组件的排列从交换格式文件中识别表结构
  + 通过计算文字水平重叠来识别列，使用pdftohtml生成的文本块
  + 通过用自底向上方式对基本内容元素进行分组的方式识别表格结构
  + 一种“soft”的基于规则的方法可以适用于不同领域
+ 数据驱动的方法
  + DeepDeSRT方法将表格结构识别任务作为图像语义分割问题，然后分别识别列和行区域
  + 一个图象转文本模型，编码表格图象然后解码表格结构为类HTML的标签序列
  尽管这个标签序列是为了表示一个表格而设计的，但它没有提供表格单元的列坐标。因此，这个标签序列不能用于恢复表格，也意味着这个模型不是一个完整的模型，所以不能作为实验基准

综上，这些方法仅仅对于简单的网格状的表格表现得较好，但对于复杂表格表现的不好
#### 方法
对于输入的PDF形式的表格，此方法通过四个步骤来识别其结构：
+ 预处理：从PDF中获取表格单元内容和关联边界（和Shigarov et al.,2016 相同）
+ 建立图：用这些表格单元构建一个无向图
+ 关联预测：用作者提出的GraphTSR模型预测邻接关系
+ 后处理：从有标签的图中恢复表格结构
##### 问题定义
假设表格中的每个表格单元可以被看作一个顶点，邻接关系可以被看作有标签的边，所以一个表格可以被表示为一个有标签的边的图  
给定顶点集V和表格T = （V, R）作为输入，问题是要得到真实关系R的一个近似
##### 图构建
使用K-近邻算法来构建E'
##### GraphTSR
GraphTSR将图的顶点和边特征作为输入，用N边到点图attention块和N点到边图attention块来分别计算他们的表示。最终它表示这些边的一个分类。
+ 顶点和边特征
  + 顶点特征：表格单元大小，绝对位置，相对位置 
  + 边特征：几种单元格距离（欧几里得距离，x轴距离和y轴距离）的绝对和相对形式，重叠单元格的x轴距离和y轴距离
+ 图attention
+ 图attention块  
  为了支持边特点，原始图被转变为带有表示原始图中边的额外标记的一式两份的图  
  用这个设置，使用N边到点图attention块和N点到边图attention块来分别编码顶点和边。这两种attention块的计算是对称的
### SciTSR数据集
数据集构建细节和统计
### 实验
