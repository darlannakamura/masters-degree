# Gaussian Noise

Comparing traditional and neural-network based methods in relation to PSNR and SSIM metrics, on Gaussian noise, with mean 0 and std = 0.1.

Experiment started at 2021/05/01 16:57:07 and ended at 2021/05/02 05:26:55, during 12 hours 29 minutes and 48 seconds.


| Methods | Dataset | Training samples | Test samples | Dimension |
|---|---|---|---|---|
| BM3D,CNN,Autoencoder,DIP,DnCNN, CGAN, K-SVD,MLP,NLM,Wiener Filter and WST | BSD300 | 6000 | 1500 | 52 X 52 |

# 1. Results

## 1.1 Table

We could see the results through the table, comparing all the methods in relation to PSNR and SSIM metrics.



| Method | PSNR (dB) | SSIM | Runtime (seconds) |
|---|---|---|---|
| K-SVD | 24.8 ± 4.93 | 0.85 ± 0.06 | 4916.94 |
| Wiener Filter | 22.28 ± 3.82 | 0.83 ± 0.07 | 1.02 |
| DIP | 23.05 ± 4.45 | 0.81 ± 0.12 | 39785.14 |
| WST | 24.71 ± 4.84 | 0.78 ± 0.13 | 1.5 |
| BM3D | 25.13 ± 5.12 | 0.77 ± 0.17 | 146.85 |
| DnCNN | 20.83 ± 5.73 | 0.77 ± 0.08 | 58.09 |
| MLP | 22.97 ± 3.85 | 0.7 ± 0.18 | 2.62 |
| NLM | 21.06 ± 3.82 | 0.68 ± 0.22 | 49.07 |
| CNN | 22.1 ± 5.58 | 0.61 ± 0.22 | 17.81 |
| Autoencoder | 25.61 ± 6.7 | 0.57 ± 0.28 | 2.41 |
| CGAN | 12.31 ± 5.20 | 0.52 ± 0.21 | 21.69 | 
| Noisy | 20.71 ± 4.32 | 0.7 ± 0.17 | --- |

## 1.2 PSNR Boxplot

![PSNR boxplot](psnr_boxplot.png)

## 1.3 SSIM Boxplot

![SSIM boxplot](ssim_boxplot.png)


## 1.4 Visual Results

![Visual results](results.png)

## 1.5 DnCNN loss

![DnCNN loss](DnCNN_loss.png)

## How to rerun the experiment?

If you want to rerun this experiment, you could use the `.metadata` directory.
This directory haves all the data, like the train and test data, the value of the PSNR methods in runtime, and data frame (pandas DataFrame) used to generate the Seaborn plots.