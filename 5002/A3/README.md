# Unsupervised Learning

In this assignment, we need to cluster a certain amount of image data without exact number of cluster.


# Overview
1. Use imagenet state-of-art model to embed images
2. Find out the best number of K for cluster
3. Use different model and strategy to cluster

### 1. Embed Images
> An implicit hypothesis in modern computer vision research[1] is that models that perform better on ImageNet necessarily perform better on other vision tasks. 
Recently, a research has been done by Google Brain, which shows that ImageNet architectures generalize well across datasets. 

Thus, it makes sense to use ResNet to embed images into vector so that I can apply unsupervised learning on images. My notebook `1103-GlanceData.ipynb` ensures both the stability and accuracy on ResNet152 across the assignment dataset by applying class **activation mapping**[2].

<table border=0 >
    <tbody>
        <tr>
            <th align="center" valign="top">  <b>Sample</b> </td>
            <th align="left" valign="top"> <b>Original Image</b></td>
            <th align="left" valign="top"> <b>Class Activation Mapping</b></td>
            <th align="left" valign="top"> <b>Guided BackPropagation</b></td>
        </tr>
        <tr>
            <td align="left" valign="center" width="25%">  <b>Index</b>: 00000 <br />   <b>Probs</b>: 0.798<br />   <b>Class</b>: {black stork, Ciconia nigra}  </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00000.jpg"> </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00000_cam.png"> </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00000_gbp.png"> </td>
        </tr>
        <tr>
            <td align="left" valign="center" width="25%">  <b>Index</b>: 00001 <br />   <b>Probs</b>: 0.709<br />   <b>Class</b>: {patio, terrace}  </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00001.jpg"> </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00001_cam.png"> </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00001_gbp.png"> </td>
        </tr>
        <tr>
            <td align="left" valign="center" width="25%">  <b>Index</b>: 00002 <br />   <b>Probs</b>: 0.443<br />   <b>Class</b>: {kite}  </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00002.jpg"> </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00002_cam.png"> </td>
            <td width="25%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_00002_gbp.png"> </td>
        </tr>
    </tbody>
</table>







### 2. Define K
Base on the research consequence above, ImageNet model (eg. ResNet, DenseNet) shall have ability to predict, or say cluster the types of images, though we don't know whether this dataset is inside ImageNet or not. During my experiment, `ResNet152` from `torchvision` return about 600 type of labels for our dataset. I would like to cluster these labels to get an approximate value of `K` at first.

Early works[3] show that the Euclidean distance (or cosine similarity) between two word vectors is an effective method for measuring the linguistic or semantic similarity of the corresponding words. So, I use to ResNet152 to predict our image dataset, following word2vec to embed the label into vector, and apply dendrogram on the word vector to figure out an approximate value of `K`. Below is the cluster result of **dendrogram** for label word vector: 

<table border=0 >
    <tbody>
        <tr>
            <td width="20%" > <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_Dendrogram500.png"> </td>
            <td width="30%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_Dendrogram200.png"> </td>
            <td width="50%"> <img src="https://github.com/sysu-zjw/MSBD-2018Fall/blob/master/img/5002A3_Dendrogram100.png"> </td>
        </tr>
    </tbody>
</table>



# References

[1] Do Better ImageNet Models Transfer Better? [[Link](https://arxiv.org/pdf/1805.08974.pdf)]

[2] Learning Deep Features for Discriminative Localization [[Link](https://arxiv.org/pdf/1512.04150.pdf)]

[3] Distributed Representations of Words and Phrases and their Compositionality [[Link](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)]