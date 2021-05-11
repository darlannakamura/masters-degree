# Gaussian Noise 3

Comparing traditional and neural-network based methods in relation to PSNR and SSIM metrics, on Gaussian noise, with mean 0 and std = 0.1.

Experiment started at 2021/05/05 23:36:31 and ended at 2021/05/06 12:08:29, during 12 hours 31 minutes and 57 seconds.


| Methods | Dataset | Training samples | Test samples | Dimension |
|---|---|---|---|---|
| BM3D,CNN,Autoencoder,DIP,DnCNN,K-SVD,MLP,NLM,Wiener Filter and WST | BSD300 | 6000 | 1500 | 52 X 52 |

# 1. Results

## 1.1 Table

We could see the results through the table, comparing all the methods in relation to PSNR and SSIM metrics.



| Method | PSNR (dB) | SSIM | Runtime (seconds) |
|---|---|---|---|
| K-SVD | 23.35 ± 4.62 | 0.84 ± 0.06 | 4923.68 |
| Wiener Filter | 21.76 ± 3.97 | 0.82 ± 0.07 | 1.02 |
| DIP | 20.12 ± 4.54 | 0.78 ± 0.13 | 39959.11 |
| WST | 25.21 ± 4.99 | 0.78 ± 0.13 | 1.46 |
| BM3D | 25.05 ± 5.05 | 0.77 ± 0.17 | 120.66 |
| DnCNN | 22.78 ± 5.44 | 0.77 ± 0.11 | 44.39 |
| MLP | 22.03 ± 3.95 | 0.7 ± 0.17 | 1.95 |
| NLM | 21.02 ± 3.88 | 0.68 ± 0.22 | 48.12 |
| CNN | 21.79 ± 4.88 | 0.6 ± 0.21 | 9.08 |
| Autoencoder | 25.69 ± 6.8 | 0.57 ± 0.28 | 1.86 |
| Noisy | 20.32 ± 4.36 | 0.7 ± 0.17 | --- |

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