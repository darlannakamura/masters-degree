import os
import math
import time
import pickle
import logging
import json

from typing import Tuple

from datetime import datetime
from time import strftime
from time import gmtime
from tqdm import tqdm

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from denoising.datasets import load_bsd300
from denoising.datasets import load_dataset
from denoising.datasets import extract_patches
from denoising.datasets import add_noise

from denoising.metrics import psnr, ssim
from denoising.utils import normalize

from report import Report
from utils import is_using_gpu

from settings import BSD300_DIR, BASE_DIR

from experiments import load_config, load_methods, Method

logging.basicConfig(level=logging.WARNING)

class Experiment:
    def __init__(self, filename:str):
        self.load_configuration(filename)

        self.methods = load_methods()
        self.methods_name = [method.name for method in self.methods]
        
        (self.x_train, self.y_train, self.x_test, self.y_test) = self.load_data()

        self.output_path = os.path.join(BASE_DIR, self.output, self.name)
        os.makedirs(self.output_path, exist_ok=True)

        self.metadata_path = os.path.join(self.output_path, ".metadata")
        os.makedirs(self.metadata_path, exist_ok=True)

    def run(self):
        self.start_date = datetime.now()
        self.train_methods()

        self.test_methods()

        self.save_results()
        self.save_metadata()
        self.generate_report()

    def load_configuration(self, filename: str):
        config = load_config(filename)
        self.__dict__.update(config)

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        imgs = load_bsd300(BSD300_DIR)
        patches = extract_patches(imgs, begin=(0,0), stride=10,
            dimension=(52,52), quantity_per_image=(5,5))
    
        y_train, y_test = load_dataset(patches, shuffle=False, split=(80,20))

        self.mean = 0.0
        self.variance = 0.01
        self.std = math.sqrt(self.variance) # 0.1

        x_train = add_noise(y_train, noise='gaussian', mean=self.mean, var=self.variance)
        x_test = add_noise(y_test, noise='gaussian', mean=self.mean, var=self.variance)

        # x_train = normalize(x_train, interval=(0,1), data_type='float')
        # x_test = normalize(x_test, interval=(0,1), data_type='float')
        
        # y_train = normalize(y_train, interval=(0,1), data_type='float')
        # y_test = normalize(y_test, interval=(0,1), data_type='float')

        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        y_train = y_train.astype('float32') / 255.0
        y_test = y_test.astype('float32') / 255.0

        return (x_train, y_train, x_test, y_test)

    def train_methods(self):
        nn_methods = []

        for method in self.methods:
            if method.need_train:
                nn_methods.append(method)

        for network in nn_methods:
            instance = network.instance
            instance.compile(**network.parameters["compile"])
            
            if "set_checkpoint" in network.parameters:
                ckpt_params = network.parameters["set_checkpoint"]

                if ckpt_params["filename"] == "default":
                    ckpt_params["filename"] = os.path.join(self.metadata_path, f'{network.name}.hdf5')
                
                instance.set_checkpoint(**ckpt_params)

            instance.fit(
                self.x_train,
                self.y_train,
                **network.parameters["fit"]
            )

            instance.save_loss_plot(os.path.join(self.output_path, f'{network.name}_loss.png'))

    def test_methods(self):        
        start_experiment_time = time.time()

        for method in tqdm(self.methods):
            start_time = time.time()
            instance = method.instance
            
            if method.is_traditional:
                predicted = instance(self.x_test, noise_std_dev=self.std)
            else:
                predicted = instance.test(self.x_test)

            method.runtime = time.time() - start_time
            method.images = predicted
            method.psnr = psnr(
                normalize(self.y_test, data_type='int'), 
                normalize(predicted, data_type='int')
            )
            method.ssim = ssim(
                normalize(self.y_test, data_type='int'), 
                normalize(predicted, data_type='int')
            )
        
        self.end_date = datetime.now()
        self.duration = time.time() - start_experiment_time

    def save_results(self):        
        self.save_visual_results()
        self.save_quantitative_results()

        self.df = self.generate_dataframe()

        self.save_psnr_boxplot()
        self.save_ssim_boxplot()

    def save_metadata(self):

        np.save(file=os.path.join(self.metadata_path, "x_train.npy"), arr=self.x_train, allow_pickle=True)
        np.save(file=os.path.join(self.metadata_path, "y_train.npy"), arr=self.y_train, allow_pickle=True)
        np.save(file=os.path.join(self.metadata_path, "x_test.npy"), arr=self.x_test, allow_pickle=True)
        np.save(file=os.path.join(self.metadata_path, "y_test.npy"), arr=self.y_test, allow_pickle=True)

        # with open(os.path.join(self.metadata_path, "methods.pickle"), 'wb') as handle:
        #     pickle.dump(self.methods, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.df.to_pickle(os.path.join(self.metadata_path, "df.pickle"))

    def get_methods_name(self):
        names = self.methods_name
        last_name = names.pop(-1)

        return f"{','.join(names)} and {last_name}"

    def generate_report(self):
        variables = {
            'experiment_name': self.name,
            'methods': self.get_methods_name(),
            'dataset': self.dataset,
            'training_samples': self.x_train.shape[0],
            'test_samples': self.x_test.shape[0],
            'dimension': f'{self.x_test.shape[1]} X {self.x_test.shape[2]}',
            'table': self.content_table,
            'about': self.about,
            'start_date': self.start_date.strftime('%Y/%m/%d %H:%M:%S'),
            'end_date': self.end_date.strftime('%Y/%m/%d %H:%M:%S'),
            'duration': strftime("%H hours %M minutes and %S seconds", gmtime(round(self.duration, 2))),
            'is_using_gpu': 'using GPU' if is_using_gpu() else 'without GPU',
        }

        report = Report(variables, template='report_template.md')
        report.to_md(os.path.join(self.output_path, 'report.md'))

    def save_visual_results(self):
        COLUMNS = 5

        # methods = self.methods
        methods = []

        for method in self.methods:
            methods.append({'name': method.name, 'images': method.images})
        
        methods.insert(0, {'name': "noisy", 'images': self.x_test})
        methods.insert(0, {'name': "ground truth", 'images': self.y_test})
        
        fig, ax = plt.subplots(COLUMNS, len(methods), figsize=(12,8))

        for i, method in enumerate(methods):
            for j in range(COLUMNS):
                ax[j,i].axis('off')

                title = method['name']
                imgs = method['images']
                
                if j == 0:
                    ax[j,i].set_title(title)
                
                ax[j,i].imshow(imgs[j, :, :, 0], cmap='gray', interpolation='nearest')
        
        fig.savefig(os.path.join(self.output_path, "results.png"), dpi=fig.dpi)

    def save_quantitative_results(self, output_type:str='markdown'):
        def order_by_ssim(e):
            return round(e.ssim.mean(), 2)

        assert output_type == 'markdown', 'output_type should be: markdown.'

        lines = [
            f"",
            "",
            "| Method | PSNR (dB) | SSIM | Runtime (seconds) |",
            "|---|---|---|---|"
        ]

        for method in sorted(self.methods, reverse=True, key=order_by_ssim):
            mean_psnr = round(method.psnr.mean(), 2)
            std_psnr = round(method.psnr.std(), 2)

            mean_ssim = round(np.mean(method.ssim), 2)
            std_ssim = round(np.std(method.ssim), 2)

            runtime = round(method.runtime, 2)

            lines.append(f"| {method.name} | {mean_psnr} ± {std_psnr} | {mean_ssim} ± {std_ssim} | {runtime} |")

        method_name = "Noisy"
        psnr_noisy = psnr(normalize(self.y_test, data_type='int'), normalize(self.x_test, data_type='int'))
        ssim_noisy = ssim(normalize(self.y_test, data_type='int'), normalize(self.x_test, data_type='int'))

        mean_psnr = round(psnr_noisy.mean(), 2)
        std_psnr = round(psnr_noisy.std(), 2) 

        mean_ssim = round(ssim_noisy.mean(), 2)
        std_ssim = round(ssim_noisy.std(), 2)

        lines.append(f"| {method_name} | {mean_psnr} ± {std_psnr} | {mean_ssim} ± {std_ssim} | --- |")

        md_text = '\n'.join(lines)

        self.content_table = md_text

        with open(os.path.join(self.output_path, "results.md"), 'w') as f:
            f.write(md_text)

    def generate_dataframe(self):
        data = []

        for method in self.methods:

            for p in method.psnr:
                data.append({
                    'metric': 'psnr',
                    'method': method.name,
                    'value': p
                })

            for s in method.ssim:
                data.append({
                    'metric': 'ssim',
                    'method': method.name,
                    'value': s
                })

        df = pd.DataFrame(data)

        return df

    def save_ssim_boxplot(self):
        self.save_boxplot(metric='ssim', filename='ssim_boxplot.png')

    def save_psnr_boxplot(self):
        self.save_boxplot(metric='psnr', filename='psnr_boxplot.png')

    def save_boxplot(self, metric: str, filename: str):
        plt.figure(figsize=(12,8))
        sns.boxplot(data=self.df[self.df['metric'] == metric], x='method', y='value', palette='Set2')
        plt.ylabel(metric.upper())

        plt.savefig(os.path.join(self.output_path, filename))

if __name__ == '__main__':
    experiment = Experiment()
    experiment.run()
