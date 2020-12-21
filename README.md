# survival_analysis
主要是用python进行生存分析的步骤，包括生存分析(逐步和单因素)，KM曲线、决策曲线，ROC曲线，训练测试样本分布比较

#step 1 KM曲线
![](https://github.com/zoufengyuan/Waveform-classification/blob/main/wave_1.png)


# Waveform-classification
根据前期125Hz的低血压数据记录和患者的基本特征数据，去预测5分钟后是否会有低血压。

# 主要思路：
首先将患者的基本信息和label提取出来，再对波形数据进行特征提取，主要的提取方法有两种：\
1、人工提取波形数据的特征：波峰的最小值、最大值、均值、标准差、RR区间的最小值、最大值、均值、标准差、偏度、峰度、R波密度等\
2、2种方法对数据进行处理，运用resnet50自动提取波形数据特征：\
①对维度为1250的波进行下采样，得到维度为1000的波，直接代入resnet模型进行特征提取，取倒数第一层完全连接的输出作为波形特征\
![](https://github.com/zoufengyuan/Waveform-classification/blob/main/wave_1.png)
![](https://github.com/zoufengyuan/Waveform-classification/blob/main/wave_2.png)

②观察到波大部分可能是呈周期性的，根据波峰的位置对波进行切割，进行下采样，最后形成维度为(15,60)的数据代入resnet进行特征提取，取倒数第一层完全连接的输出作为波形特征\
![](https://github.com/zoufengyuan/Waveform-classification/blob/main/wave_cnn_1.png)
![](https://github.com/zoufengyuan/Waveform-classification/blob/main/wave_cnn_2.png)
# 脚本框架：
![](https://github.com/zoufengyuan/Waveform-classification/blob/main/%E7%A8%8B%E5%BA%8F%E6%A1%86%E6%9E%B6%E5%9B%BE.png)

model_cnn.py等含cnn标识的脚本为维度为(15,60)的数据代入resnet的脚本，cnn_data_reshape.py是对波进行切割处理的脚本

# 预测结果：
总体预测准确率为0.75左右,有待提升
