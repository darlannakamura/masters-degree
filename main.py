import argparse
import multiprocessing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run your denoising experiments.')
    parser.add_argument('-f', '--file', type=str, required=True, help='experiment configuration file')
    parser.add_argument('-p', '--phase', type=str, default='test', choices=('train', 'test', 'compare'), help='experiment phase to run')
    parser.add_argument('-t', '--test', type=bool, default=False, help='test your experiment')
    parser.add_argument('-m', '--method', type=str, required=False, help='method to be compared')
    parser.add_argument('-k', type=int, required=False, help='iteration of K-fold')

    args = parser.parse_args()

    if args.phase == 'train':
        from train import train
        train(args.file, args.test)
    elif args.phase == 'test':
        multiprocessing.set_start_method('spawn', force=True)
        from experiments import Experiment

        e = Experiment(filename=args.file, test=args.test, k=args.k)
        e.run()
    elif args.phase == 'compare':
        from compare_method_parameters import compare
        compare(args.file, args.method, check=False)
