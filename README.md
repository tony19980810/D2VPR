# D2-VPR: A Parameter-efficient Visual-foundation-model-based Visual Place Recognition Method via Knowledge Distillation and Deformable Aggregation



Official repository for the paper "D2-VPR: A Parameter-efficient Visual-foundation-model-based Visual Place Recognition Method via Knowledge Distillation and Deformable Aggregation" 

## Abstract
Visual Place Recognition (VPR) aims to determine the geographic location of a query image by retrieving its most visually similar counterpart from a geo-tagged reference database. Recently, the emergence of the powerful visual foundation model, DINOv2, trained in a self-supervised manner on massive datasets, has significantly improved VPR performance. This improvement stems from DINOv2’s exceptional feature generalization capabilities but is often accompanied by increased model complexity and computational overhead that impede deployment on resource-constrained devices. To address this challenge, we propose D2-VPR, a Distillation- and Deformable-based framework that retains the strong feature extraction capabilities of visual foundation models while significantly reducing model parameters and achieving a more favorable performance-efficiency trade-off. Specifically, first, we employ a two-stage training strategy that integrates knowledge distillation and fine-tuning. Additionally, we introduce a Distillation Recovery Module (DRM) to better align the feature spaces between the teacher and student models, thereby minimizing knowledge transfer losses to the greatest extent possible. Second, we design a Top-Down-attention-based Deformable Aggregator (TDDA) that leverages global semantic features to dynamically and adaptively adjust the Regions of Interest (ROI) used for aggregation, thereby improving adaptability to irregular structures. Extensive experiments demonstrate that our method achieves competitive performance compared to state-of-the-art approaches. Meanwhile, it reduces the parameter count by approximately 64.2% and FLOPs by about 62.6% (compared to CricaVPR).

## 📄Paper

Main Paper: [link](https://arxiv.org/abs/2511.12528)

## 🛠️Setup

This repo follows the framework of [CricaVPR](https://github.com/Lu-Feng/CricaVPR/tree/main) and [BoQ](https://github.com/amaralibey/Bag-of-Queries). 

We use [GSV-Cities](https://github.com/amaralibey/gsv-cities) for training, and the [Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) for evaluation. You can download the GSV-Cities datasets [HERE](https://www.kaggle.com/datasets/amaralibey/gsv-cities), and refer to [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) to prepare test datasets.

The test dataset should be organized in a directory tree as such:
```bash
├── datasets_vg
    └── datasets
        └── pitts30k
            └── images
                ├── train
                │   ├── database
                │   └── queries
                ├── val
                │   ├── database
                │   └── queries
                └── test
                    ├── database
                    └── queries
```


## 🏋️‍♂Training

Before we begin, please download the CricaVPR weights [LINK](https://drive.google.com/file/d/171lCcxZFFnvEvo88ntIwELeBegcMTEJs/view) and Dinov2-small weights [LINK](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth).

Please use the following code for distillation:

```bash
python distillation.py
```
Before starting fine-tuning, please download the [msls-val](https://drive.google.com/file/d/1zApxUkwq4E_1gMaiDAhzorJ6DO2G07fr/view?usp=drive_link) and [pitts30k-val](https://drive.google.com/file/d/184Scho9iQ1Dg3kjXSYAIscWliF5n8Ty4/view?usp=drive_link) validation datasets for testing and update the dataset paths in the finetune.py accordingly. Then select the best distillation weights, change the corresponding path in finetune_script.py and use the following code for finetune:

```bash
python finetune.py
```

## 🔮 Model Weights 

The model (no cross-image encoder).
<table>
<thead>
  <tr>
    <th rowspan="2">DOWNLOAD<br></th>
    <th colspan="3">Pitts30k</th>
    <th colspan="3">MSLS-val</th>
    <th colspan="3">SPED</th>
    <th colspan="3">Pitts250k</th>
    <th colspan="3">AmsterTime</th>
    <th colspan="3">Nordland</th>
  </tr>
  <tr>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th rowspan="3"><a href="https://drive.google.com/file/d/1uh4XZsmRBhPkGuBNTWtjNypoGBd2JWyy/view?usp=drive_link">LINK</a></th>
    <td>91.7</td>
    <td>95.8</td>
    <td>96.8</td>
    <td>90.7</td>
    <td>95.4</td>
    <td>96.4</td>
    <td>86.0</td>
    <td>92.9</td>
    <td>94.1</td>
    <td>94.4</td>
    <td>98.3</td>
    <td>98.9</td>
    <td>49.1</td>
    <td>70.7</td>
    <td>76.1</td>
    <td>77.1</td>
    <td>88.6</td>
    <td>91.9</td>
  </tr>
</tbody>
</table>


The distillation model (cross-image encoder).
<table>
<thead>
  <tr>
    <th rowspan="2">DOWNLOAD<br></th>
    <th colspan="3">Pitts30k</th>
    <th colspan="3">SPED</th>
    <th colspan="3">AmsterTime</th>
    <th colspan="3">Pitts250k</th>
  </tr>
  <tr>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th rowspan="3"><a href="https://drive.google.com/file/d/1tcsn-U-v-jfRHr9-BIyIyCfezdse_iy_/view?usp=drive_link">LINK</a></td>
    <td>94.9</td>
    <td>97.3</td>
    <td>98.0</td>
    <td>90.9</td>
    <td>96.0</td>
    <td>97.2</td>
    <td>62.9</td>
    <td>80.8</td>
    <td>85.3</td>
    <td>97.8</td>
    <td>99.4</td>
    <td>99.7</td>
  </tr>
</tbody>
</table>

The fine-tuned model (cross-image encoder).
<table>
<thead>
  <tr>
    <th rowspan="2">DOWNLOAD<br></th>
    <th colspan="3">MSLS-val</th>
    <th colspan="3">Nordland</th>
    
  </tr>
  <tr>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
    <th>R@1</th>
    <th>R@5</th>
    <th>R@10</th>
   
  </tr>
</thead>
<tbody>
  <tr>
    <th rowspan="3"><a href="https://drive.google.com/file/d/1FwJY2hhuMLh4OJFTr7cMzRM7WrZ2Qb9a/view?usp=drive_link">LINK</a></td>
    <td>91.6</td>
    <td>96.1</td>
    <td>97.2</td>
    <td>86.6</td>
    <td>94.1</td>
    <td>96.0</td>
  </tr>
</tbody>
</table>

## 🧩Evaluation

Pleas download the model weights (no cross-image encoder) into the directory, and use the following code for evaluation:
```bash
python eval_no_encoder.py
```

Pleas download the distillation and finetune weights (cross-image encoder) into the directory, and use the following code for evaluation:
```bash
python eval_encoder.py
```

##  🙏Acknowledgements


The framework of the code are based on the excellent work of [CricaVPR](https://github.com/Lu-Feng/CricaVPR/tree/main) and [BoQ](https://github.com/amaralibey/Bag-of-Queries). The experiments are built upon the excellent work of [CricaVPR](https://github.com/Lu-Feng/CricaVPR/tree/main), [CosPlace](https://github.com/gmberton/CosPlace), [EigenPlaces](https://github.com/gmberton/EigenPlaces), [MixVPR](https://github.com/amaralibey/MixVPR),  [BoQ](https://github.com/amaralibey/Bag-of-Queries), [SALAD](https://github.com/serizba/salad), [SuperVLAD](https://github.com/Lu-Feng/SuperVLAD), [SelaVPR](https://github.com/Lu-Feng/SelaVPR), [FoL](https://github.com/chenshunpeng/FoL) and [EDTformer](https://github.com/Tong-Jin01/EDTformer). The dataset is composed based on the excellent work of [gsv-cities](https://github.com/amaralibey/gsv-cities), [deep-visual-geo-localization-benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) and [VPR-datasets-downloader
](https://github.com/gmberton/VPR-datasets-downloader). We would like to express our gratitude for their open-source efforts.


## 📚 Citation
If you find this repo useful for your research, please cite the paper

```
@article{zhang2025d,
  title={D\textsuperscript{2}-VPR: A Parameter-efficient Visual-foundation-model-based Visual Place Recognition Method via Knowledge Distillation and Deformable Aggregation},
  author={Zhang, Zheyuan and Zhang, Jiwei and Zhou, Boyu and Duan, Linzhimeng and Chen, Hong},
  journal={arXiv preprint arXiv:2511.12528},
  year={2025}
}

```
