# Topic :

Real Time Traffic Management Using Machine Learning

# Group Members :

Jyoti Tiwari

Ankita Deshmukh

Gayatri Godepure

# Brief Description :

<p align="justify">
People in today’s era usually have tendency of using their own private vehicles for commutation rather than using public or pooled means of transport and this results in large number of private vehicles on road. This endless increasing number of vehicles on road gives rise to many problems amongst them traffic congestion tops in every aspect. In such scenario one cannot restrict individual to limit the usage of their private vehicles but what we can do is at least manage traffic flow in a way that it doesnot alleviate congestion issues.
</p>

<p align="justify">
There are many projects emerging in order to convert the current transport system of cities to ‘Smart system’ and there are many initiatives under this, one of this is Intelligent Transport System. Many initiatives were taken to design a system that can perform realtime monitoring of traffic signals i.e. the traffic signal switching time will not be predefined o n e, instead the switching time will depend on the count of vehicles on each side of the road. This process of getting the count of vehicle on the road can be achieved using various detection techniques.
</p>

<p align="justify">
Our aim is to design and develop a miniature to depict the current road situation along with monitoring and handling the traffic issues. Hence to proceed with this project we are using a pretrained model YOLO to perform the task of object detection. The pretrained model YOLO uses OpenCV for object detection along with multiple foreground and background subtraction and removal of noise from the input image. The CCTV cameras that are being used for surveillance purpose can be made use to capturing the footage of the road, this image will be passed to the pretrained model as input image. To do so each side of the road will be divided into particular frames of same height and width for capturing the image. The count obtained from the image is the fed to the Raspberry board. As per the count obtained, switching time will be assigned for each side of road. The program will initially check if the count of vehicle in all frame is approximately same then the switching will remain at its predefined regular interval for all sides of signal, the real-time switching for the signal will be performed if the count of vehicles in all frames varies as threshold difference which be provided.
</p>

# Note :

Yolo-Coco Weights File is Not Attached from this folder.

But This Weights file is important to run this Project.
