# Gaussian Noise 2

Comparing traditional and neural-network based methods in relation to PSNR and SSIM metrics, on Gaussian noise, with mean 0 and std = 0.1.

Experiment started at 2021/05/04 18:54:03 and ended at 2021/05/05 07:10:49, during 12 hours 16 minutes and 45 seconds.


| Methods | Dataset | Training samples | Test samples | Dimension |
|---|---|---|---|---|
| BM3D,CNN,Autoencoder,DIP,DnCNN,K-SVD,MLP,NLM,Wiener Filter and WST | BSD300 | 6000 | 1500 | 52 X 52 |

# 1. Results

## 1.1 Table

We could see the results through the table, comparing all the methods in relation to PSNR and SSIM metrics.



| Method | PSNR (dB) | SSIM | Runtime (seconds) |
|---|---|---|---|
| K-SVD | 23.6 ± 4.76 | 0.84 ± 0.06 | 4939.92 |
| Wiener Filter | 21.63 ± 3.91 | 0.82 ± 0.07 | 0.99 |
| DIP | 20.67 ± 4.51 | 0.79 ± 0.13 | 39026.65 |
| WST | 23.78 ± 5.05 | 0.78 ± 0.13 | 1.44 |
| BM3D | 25.53 ± 5.11 | 0.77 ± 0.17 | 118.87 |
| DnCNN | 23.82 ± 4.75 | 0.75 ± 0.12 | 44.64 |
| NLM | 20.82 ± 3.84 | 0.68 ± 0.22 | 48.06 |
| CNN | 21.79 ± 4.77 | 0.63 ± 0.2 | 14.45 |
| Autoencoder | 25.9 ± 7.12 | 0.57 ± 0.28 | 2.06 |
| MLP | 8.32 ± 4.81 | 0.0 ± 0.02 | 1.98 |
| Noisy | 20.87 ± 4.3 | 0.7 ± 0.17 | --- |

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