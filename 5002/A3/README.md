# Unsupervised Learning

In this assignment, we need to cluster a certain amount of image data without exact number of cluster.


# Overview
1. Use imagenet state-of-art model to embed images
2. Find out the best value of K for cluster
3. Use different model and strategy to cluster

### 1. Embed Images
> An implicit hypothesis in modern computer vision research[1] is that models that perform better on ImageNet necessarily perform better on other vision tasks. 
Recently, a research has been done by Google Brain, which shows that ImageNet architectures generalize well across datasets. 

Though we have reference to support this assumption, we still need to evaluate the predicting ability for ResNet on this dataset.
My notebook `1103-GlanceData.ipynb` apply **Class Activation Mapping**[2] and **Guided BackPropagation**[3] to ensure the stability and accuracy on ResNet152 across the assignment dataset.

<table border=0 >
    <tbody>
        <tr>
            <th align="center" valign="center">  <b>Sample</b> </th>
            <th align="center" valign="center"> <b>Original Image</b></th>
            <th align="center" valign="center"> <b>Class Activation Mapping</b></th>
            <th align="center" valign="center"> <b>Guided BackPropagation</b></th>
        </tr>
        <tr>
            <td align="left" valign="center" width="25%">  <b>Index</b>: 00000 <br />   <b>Probs</b>: 0.798<br />   <b>Class </b>: {black stork, Ciconia nigra}  </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00000.jpg"> </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00000_cam.png"> </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00000_gbp.png"> </td>
        </tr>
        <tr>
            <td align="left" valign="center" width="25%">  <b>Index</b>: 00001 <br />   <b>Probs</b>: 0.709<br />   <b>Class </b>: {patio, terrace}  </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00001.jpg"> </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00001_cam.png"> </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00001_gbp.png"> </td>
        </tr>
        <tr>
            <td align="left" valign="center" width="25%">  <b>Index</b>: 00002 <br />   <b>Probs</b>: 0.443<br />   <b>Class </b>: {kite}  </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00002.jpg"> </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00002_cam.png"> </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00002_gbp.png"> </td>
        </tr>
    </tbody>
</table>



Some network visualization technologies(eg. cam and gbp) are taken as safety belt to ensure the knowledge transfer.



### 2. Selection of K
Base on the research consequence above, ImageNet model (eg. ResNet, DenseNet) shall have ability to predict, or say cluster different types of images, even though the dataset is not inside ImageNet. During my experiment, `ResNet152` from `torchvision` return about 600 types of labels for our dataset. I would like to cluster these labels to get an approximate value of `K` at first.

Early works[4] show that the Euclidean distance (or cosine similarity) between two word vectors is an effective method for measuring the linguistic or semantic similarity of the corresponding words. So, I use ResNet152 to predict our image dataset, following word2vec to embed the label into vector, and apply dendrogram on the word vector to figure out an approximate value of `K`. Below is the cluster result of **dendrogram** for label word vector: 

<table border=0 >
    <tbody>
        <tr>
            <td width="20%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_Dendrogram500.png"> </td>
            <td width="30%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_Dendrogram200.png"> </td>
            <td width="50%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_Dendrogram100.png"> </td>
        </tr>
    </tbody>
</table>

I will choose the inflection point in dendrogram where **K=20**

### 3. K-means clustering
I use K-means to cluster the 5011 samples each with 2048 features embedded by `ResNet152`. 

I also do random sampling and try to interpret the meaning of each cluster. Results are shown below:
### 11.12 Update
<table border=0 >
    <tbody>
        <tr>
            <th align="center" valign="center" width="10%">  <b>Cluster</b> </th>
            <th align="center" valign="center" width="15%"> <b>Sample1</b></th>
            <th align="center" valign="center" width="15%"> <b>Sample2</b></th>
            <th align="center" valign="center" width="15%"> <b>Sample3</b></th>
            <th align="center" valign="center" width="15%"> <b>Sample4</b></th>
            <th align="center" valign="center" width="15%"> <b>Sample5</b></th>
            <th align="center" valign="center" width="15%"> <b>Interpret</b></th>
        </tr>
        <tr>
            <td align="left" valign="center" width="10%">  <b> 0 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label0_Image2060.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label0_Image2579.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label0_Image3264.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label0_Image4673.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label0_Image1248.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Horse </b>
        </tr>
        <tr>
            <td align="left" valign="center" width="10%">  <b> 1 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label1_Image3706.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label1_Image4938.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label1_Image1311.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label1_Image1164.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label1_Image1999.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Bus </b>
        </tr>
        <tr>
            <td align="left" valign="center" width="10%">  <b> 2 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label2_Image2677.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label2_Image3504.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label2_Image802.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label2_Image2026.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label2_Image3615.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Human </b>
        </tr>
        <tr>
            <td align="left" valign="center" width="10%">  <b> 3 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label3_Image4852.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label3_Image4297.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label3_Image3100.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label3_Image4053.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label3_Image4270.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Plane </b>
        </tr>
                <tr>
            <td align="left" valign="center" width="10%">  <b> 4 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label4_Image2639.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label4_Image1949.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label4_Image879.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label4_Image4773.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label4_Image4345.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Ship </b>
        </tr>
                <tr>
            <td align="left" valign="center" width="10%">  <b> 5 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label5_Image2028.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label5_Image4614.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label5_Image2051.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label5_Image168.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label5_Image3916.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Table </b>
        </tr>
                <tr>
            <td align="left" valign="center" width="10%">  <b> 6 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label6_Image809.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label6_Image4074.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label6_Image4097.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label6_Image3949.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label6_Image3335.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Motorcycle </b>
        </tr>
                        <tr>
            <td align="left" valign="center" width="10%">  <b> 7 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label7_Image2178.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label7_Image3426.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label7_Image1333.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label7_Image3923.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label7_Image2950.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Chair </b>
        </tr>
                        <tr>
            <td align="left" valign="center" width="10%">  <b> 8 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label8_Image1500.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label8_Image3987.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label8_Image729.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label8_Image4342.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label8_Image79.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> ? </b>
        </tr>
                        <tr>
            <td align="left" valign="center" width="10%">  <b> 9 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label9_Image451.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label9_Image2148.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label9_Image3478.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label9_Image3863.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label9_Image1677.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Screen </b>
        </tr>
                        <tr>
            <td align="left" valign="center" width="10%">  <b> 10 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label10_Image2370.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label10_Image1912.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label10_Image4546.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label10_Image3743.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label10_Image2287.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Plant </b>
        </tr>
                                <tr>
            <td align="left" valign="center" width="10%">  <b> 11 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label11_Image4068.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label11_Image3904.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label11_Image4002.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label11_Image1664.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label11_Image3721.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> ? </b>
        </tr>
                                <tr>
            <td align="left" valign="center" width="10%">  <b> 12 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label12_Image4838.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label12_Image4405.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label12_Image3399.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label12_Image386.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label12_Image2869.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Train </b>
        </tr>
                               <tr>
            <td align="left" valign="center" width="10%">  <b> 13 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label13_Image2511.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label13_Image2046.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label13_Image3774.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label13_Image486.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label13_Image4376.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Bike </b>
        </tr>
                               <tr>
            <td align="left" valign="center" width="10%">  <b> 14 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label14_Image510.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label14_Image3604.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label14_Image2058.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label14_Image3423.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label14_Image1490.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Dog </b>
        </tr>
                               <tr>
            <td align="left" valign="center" width="10%">  <b> 15 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label15_Image1070.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label15_Image1732.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label15_Image4728.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label15_Image2134.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label15_Image635.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Dog again </b>
        </tr>
                               <tr>
            <td align="left" valign="center" width="10%">  <b> 16 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label16_Image4834.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label16_Image2005.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label16_Image1945.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label16_Image4030.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label16_Image4389.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Car </b>
        </tr>
                               <tr>
            <td align="left" valign="center" width="10%">  <b> 17 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label17_Image2043.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label17_Image120.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label17_Image4581.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label17_Image3982.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label17_Image760.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Sheep/Cow </b>
        </tr>
                               <tr>
            <td align="left" valign="center" width="10%">  <b> 18 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label18_Image600.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label18_Image3051.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label18_Image4683.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label18_Image4931.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label18_Image2398.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Bike </b>
        </tr>
                                <tr>
            <td align="left" valign="center" width="10%">  <b> 19 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label19_Image3411.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label19_Image2637.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label19_Image392.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label19_Image3266.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3/5002A3_label19_Image4242.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Cat </b>
        </tr>
    </tbody>
</table>









<table border=0 >
    <tbody>
        <tr>
            <th align="center" valign="center" width="10%">  <b>Cluster</b> </th>
            <th align="center" valign="center" width="15%"> <b>Sample1</b></th>
            <th align="center" valign="center" width="15%"> <b>Sample2</b></th>
            <th align="center" valign="center" width="15%"> <b>Sample3</b></th>
            <th align="center" valign="center" width="15%"> <b>Sample4</b></th>
            <th align="center" valign="center" width="15%"> <b>Sample5</b></th>
            <th align="center" valign="center" width="15%"> <b>Interpret</b></th>
        </tr>
        <tr>
            <td align="left" valign="center" width="10%">  <b> 0 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label0_Image558.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label0_Image1579.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label0_Image2134.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label0_Image2782.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label0_Image4603.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Dog </b>
        </tr>
        <tr>
            <td align="left" valign="center" width="10%">  <b> 1 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label1_Image637.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label1_Image1335.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label1_Image1731.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label1_Image2093.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label1_Image2152.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Chair </b>
        </tr>
        <tr>
            <td align="left" valign="center" width="10%">  <b> 2 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label2_Image382.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label2_Image668.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label2_Image2667.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label2_Image2789.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label2_Image4706.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Train </b>
        </tr>
        <tr>
            <td align="left" valign="center" width="10%">  <b> 3 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label3_Image886.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label3_Image982.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label3_Image2422.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label3_Image3803.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label3_Image4923.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Plant </b>
        </tr>
                <tr>
            <td align="left" valign="center" width="10%">  <b> 4 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label4_Image356.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label4_Image1923.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label4_Image3427.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label4_Image3552.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label4_Image3678.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Airplane </b>
        </tr>
                <tr>
            <td align="left" valign="center" width="10%">  <b> 5 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label5_Image1535.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label5_Image3523.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label5_Image3742.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label5_Image4817.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label5_Image4872.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Car </b>
        </tr>
                <tr>
            <td align="left" valign="center" width="10%">  <b> 6 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label6_Image1504.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label6_Image1820.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label6_Image2291.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label6_Image2989.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label6_Image4466.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Cow/Sheep </b>
        </tr>
                        <tr>
            <td align="left" valign="center" width="10%">  <b> 7 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label7_Image598.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label7_Image2442.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label7_Image2932.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label7_Image3029.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label7_Image3748.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Bus </b>
        </tr>
                        <tr>
            <td align="left" valign="center" width="10%">  <b> 8 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label8_Image228.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label8_Image1486.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label8_Image2767.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label8_Image3374.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label8_Image4236.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Motorcycle </b>
        </tr>
                        <tr>
            <td align="left" valign="center" width="10%">  <b> 9 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label9_Image2049.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label9_Image3014.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label9_Image3081.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label9_Image4275.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label9_Image4447.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Human </b>
        </tr>
                        <tr>
            <td align="left" valign="center" width="10%">  <b> 10 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label10_Image407.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label10_Image1284.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label10_Image2520.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label10_Image3371.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label10_Image3926.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Dog??? </b>
        </tr>
                                <tr>
            <td align="left" valign="center" width="10%">  <b> 11 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label11_Image99.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label11_Image2176.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label11_Image2427.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label11_Image3787.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label11_Image4233.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Computer </b>
        </tr>
                                <tr>
            <td align="left" valign="center" width="10%">  <b> 12 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label12_Image211.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label12_Image2629.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label12_Image4192.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label12_Image4432.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label12_Image4586.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Ship </b>
        </tr>
                               <tr>
            <td align="left" valign="center" width="10%">  <b> 13 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label13_Image239.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label13_Image1643.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label13_Image1977.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label13_Image2908.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label13_Image3535.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Table </b>
        </tr>
                               <tr>
            <td align="left" valign="center" width="10%">  <b> 14 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label14_Image353.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label14_Image509.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label14_Image2833.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label14_Image3946.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label14_Image4745.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Horse </b>
        </tr>
                               <tr>
            <td align="left" valign="center" width="10%">  <b> 15 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label15_Image14.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label15_Image2254.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label15_Image3138.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label15_Image3881.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label15_Image4824.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Human??? </b>
        </tr>
                               <tr>
            <td align="left" valign="center" width="10%">  <b> 16 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label16_Image801.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label16_Image1146.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label16_Image2285.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label16_Image3867.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label16_Image4498.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Car?? </b>
        </tr>
                               <tr>
            <td align="left" valign="center" width="10%">  <b> 17 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label17_Image315.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label17_Image953.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label17_Image1934.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label17_Image4034.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label17_Image4843.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Cat </b>
        </tr>
                               <tr>
            <td align="left" valign="center" width="10%">  <b> 18 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label18_Image739.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label18_Image3700.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label18_Image3718.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label18_Image4367.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label18_Image4395.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Bike </b>
        </tr>
                                <tr>
            <td align="left" valign="center" width="10%">  <b> 19 </b>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label19_Image384.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label19_Image922.jpg"> </td>
            <td width="15%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label19_Image1508.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label19_Image1748.jpg"> </td>
            <td width="15%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_label19_Image3428.jpg"> </td>
            <td align="left" valign="center" width="15%">  <b> Bird </b>
        </tr>
    </tbody>
</table>




# References

[1] Do Better ImageNet Models Transfer Better? [[Link](https://arxiv.org/pdf/1805.08974.pdf)]

[2] Learning Deep Features for Discriminative Localization [[Link](https://arxiv.org/pdf/1512.04150.pdf)]

[3] STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET [[Link](https://arxiv.org/pdf/1412.6806.pdf)]

[4] Distributed Representations of Words and Phrases and their Compositionality [[Link](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)]