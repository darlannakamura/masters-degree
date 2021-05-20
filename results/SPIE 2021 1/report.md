# SPIE 2021 1

Comparing traditional and neural-network bas ed methods in relation to PSNR and SSIM metrics, on SPIE 2021 dataset.

Experiment started at 2021/05/20 12:21:37 and ended at 2021/05/20 12:39:43, during 00 hours 18 minutes and 05 seconds.


| Methods | Dataset | Training samples | Test samples | Dimension |
|---|---|---|---|---|
| BM3D,CGAN,CNN,DnCNN,DnCNN10,MLP,NLM,Wiener Filter and WST | spie_2021 | 15000 | 3750 | 50 X 50 |

# 1. Results

## 1.1 Table

We could see the results through the table, comparing all the methods in relation to PSNR and SSIM metrics.



| Method | PSNR (dB) | SSIM | Runtime (seconds) |
|---|---|---|---|
| DnCNN10 | 27.08 ± 1.01 | 0.69 ± 0.03 | 94.36 |
| DnCNN | 25.94 ± 1.95 | 0.64 ± 0.03 | 196.05 |
| BM3D | 27.13 ± 0.73 | 0.61 ± 0.04 | 567.92 |
| Wiener Filter | 22.18 ± 2.54 | 0.59 ± 0.05 | 2.46 |
| CNN | 23.45 ± 1.39 | 0.56 ± 0.03 | 52.61 |
| WST | 24.87 ± 0.69 | 0.56 ± 0.03 | 3.16 |
| CGAN | 25.22 ± 1.12 | 0.55 ± 0.05 | 39.56 |
| NLM | 24.57 ± 1.36 | 0.46 ± 0.04 | 108.43 |
| MLP | 8.85 ± 2.92 | 0.0 ± 0.0 | 8.34 |
| Noisy | 18.96 ± 1.83 | 0.3 ± 0.04 | --- |

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