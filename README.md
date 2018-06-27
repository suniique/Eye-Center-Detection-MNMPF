## Pre Requisites

- Python 3+

- OpenCV 2+

- NumPy

  ​



### MNMPF.py

areaMean计算图像最小邻域均值

projectionX和projectionY计算x，y方向投影，返回沿着某一个轴的最小值

MNMPF计算两个轴的投影的波谷，返回最小值作为眼球中心

pupilBorder对两个轴做积分投影限制探测范围，但是现在没用。



###libdetection.py

利用opencv获取摄像头

利用dlib探测人脸和特征点，分割出人眼

扩展我们探测到的区域，做过一些处理后利用MNMPF计算出中心

将眼眶中心点和探测到的中心连线（空间直线），计算出它们的交点（如果两条直线异面，就计算公垂线的中点），将其投影到摄像机平面上，作为视线交点。



### server.py

搭建一个简单websocket服务器，解析请求头，由服务端发送消息给前端，从而执行翻页。（代码有点复杂不细说了）。



