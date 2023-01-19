# A Weakly Supervised Semi-automatic Image Labeling Approach for Deformable Linear Objects

### Link to IEEE Xplore :page_with_curl:
[paper](https://ieeexplore.ieee.org/document/10008018) 

### Abstract

The presence of Deformable Linear Objects (DLOs) such as wires, cables or ropes in our everyday life is massive. However, the applicability of robotic solutions to DLOs is still marginal due to the many challenges involved in their perception. In this paper, a methodology to generate datasets from a mixture of synthetic and real samples for the training of DLOs segmentation approaches is thus presented. The method is composed of two steps. First, key-points along a real-world DLO are labeled employing a VR tracker operated by a user. Second, synthetic and real-world datasets are mixed for the training of semantic and instance segmentation deep learning algorithms to study the benefit of real-world data in DLOs segmentation. To validate this method a user study and a parameter study are conducted. The results show that the VR tracker labeling is usable as other labeling technique but reduces the number of clicks. Moreover, mixing real-world and synthetic DLOs data can improve the IoU score of a semantic segmentation algorithm by circa 5%. Therefore, this work demonstrates that labeling real-world data via a VR tracker can be done quickly and, if the real-world data are mixed with synthetic data, performances of segmentation algorithms for DLOs can be improved.


### How to run

Download associated data and model weights from [here](https://mega.nz/file/VAl1UT4R#5UuvXU8-g-S3J9R_z9w_cR_Yjxh_oBjy5kuRyZ8XOjs).

Change the paths inside ```run.py``` and launch the pixel-wise DLOs instance masks generator with:
  ```
  python run.py
  ```
 
