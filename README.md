# 表格检测(Table Detection)
### 从一个页面中检测出表格所在的区域
<>计算机视觉中的目标检测任务  
  

# 表格结构识别(Table Structure Recognition)
### 在检测到的表格区域的基础上，进一步将表格的内容与逻辑结构识别出来
<>早期的表格识别研究主要是基于**启发式规则**的方法，既有基于图像文档的方法，也有基于PDF文档的方法。例如由Kieninger等人提出的T-Rect系统使用**自底向上**的方法对文档图像进行**连通分支**分析，然后按照定义的规则进行合并，得到逻辑文本块。而之后由Yildiz等人提出的pdf2table则是第一个在PDF文档上进行表格识别的方法，它利用了PDF文件的一些**特有信息**（例如文字、绘制路径等图像文档中难以获取的信息）来协助表格识别。而在最近的工作中，Koci等人将页面中的布局区域表示为图（Graph）的形式，然后使用了Remove and Conquer(RAC)算法从中将表格作为一个**子图识别**出来。
  
    
**表格结构识别任务**  
(1)A Genetic-based Search for Adaptive Table Recognition in Spreadsheets:  
电子表格:遗传算法、预设种子以及噪声训练数据  
  
(2)Table Row Segmentation:  
手写表格:可能的行分隔符候选项>正确的候选项  
  
(3)**Deep Splitting and Merging for Table Structure Decomposition:**  
先自顶向下、再自底向上的两阶段表格结构识别方法SPLERGE  
分为Split和Merge两个部分  
  
(4)DeepTabStr:Deep Learning based Table Structure Recognition:  
引入变形卷积的概念  
将表格结构检测视为一个对象检测问题，将表格的行和列当做是要检测的对象  
  
(5)ReS2TIM: Reconstruct SyntacticStructures from Table Images:  
表格重建工作  
使用单元格关系判别网络判断任意两个单元格的相邻关系  
  
(6)Rethinking Semantic Segmentationfor Table Structure Recognition in Documents:
将表格结构的识别定义为语义分割问题  
FCN: Encoder-Decoder  
  
(7)Rethinking Table Recognitionusing Graph Neural Networks:  
GNN  
  
(8)TableStructure Extraction with Bi-directional Gated Recurrent Unit Networks:  
单元格在行列上具有重复性的序列特征->循环神经网络  
  
(9)TableNet: Deep Learning Model for End-to-end Table Detection and Tabular Data Extraction from Scanned Document Images  
编码器阶段使用了ImageNet上预训练的VGG-19模型来提取特征  
解码器阶段分成两个分支，分别上采样恢复到原图大小并最终得到表格和表格区域中列分割的mask图
  
  
**问题**  
高精度（高IoU阈值）的表格区域检测任务仍然还有性能提升的余地  
表格结构识别任务性能提升空间  
表格结构识别任务的研究，需要包含大量的标注数据的可靠数据集  
