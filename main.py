import argparse
import multiprocessing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run your denoising experiments.')
    parser.add_argument('-f', '--file', type=str, required=True, help='experiment configuration file')
    parser.add_argument('-p', '--phase', type=str, default='test', choices=('train', 'test'), help='experiment phase to run')
    parser.add_argument('-t', '--test', type=bool, default=False, help='test your experiment')

    args = parser.parse_args()

    if args.phase == 'train':
        from train import train
        train(args.file, args.test)
    elif args.phase == 'test':
        multiprocessing.set_start_method('spawn', force=True)
        from experiments import Experiment

        e = Experiment(filename=args.file, test=args.test)
        e.run()
