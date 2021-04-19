from experiments import Experiment
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run your denoising experiments.')
    parser.add_argument('-f', '--file', type=str, required=True, help='experiment configuration file')

    args = parser.parse_args()

    e = Experiment(filename=args.file)
    e.run()
