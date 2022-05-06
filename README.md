### 《数字图像处理》课程设计

#### 目标任务

1、将图片中的信息提取为表格；
提取“姓名”、“准考证号”、“已录取”，保存为新表格
2、在已有表格中查找指定数据并进行标注。
将“已录取”数据增加到原有表格中

#### 思路步骤

* 1.使用灰度方法读取图片，并利用局部阈值分割方法分割为像素只有0和255的图片
* 2.使用基于图形学腐蚀操作和膨胀操作，分别识别横线和竖线
* 3.找出横线和竖线的交点，并把交点像素的位置保存起来
* 4.通过图片相减得到纯文字的图片，根据像素位置来分割图片
* 5.对分割好的图片使用cnocr文字识别识别中文，使用tesseract识别数字，并将其保存到数组中
* 6.将书组中的内容分别放入新的excel表格并保存，并根据名字信息对原有表格进行修改

#### 详细说明及结果
见[博客地址](https://caixiongjiang.github.io/blog/2022/%E6%95%B0%E5%AD%97%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86/%E6%95%B0%E5%AD%97%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86%E8%AF%BE%E7%A8%8B%E8%AE%BE%E8%AE%A1%E8%A1%A8%E6%A0%BC%E5%9B%BE%E7%89%87%E6%96%87%E5%AD%97%E8%AF%86%E5%88%AB/)