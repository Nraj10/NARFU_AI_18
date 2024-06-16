from .processor import Processor
import argparse

if(__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--crop_name')
    parser.add_argument('--layout_name')
    args = parser.parse_args()
    if(args.layout_name):
        processor = Processor([args.layout_name])
    else:
        processor = Processor()
    if(args.crop_name):
        processor.process_via_script(args.crop_name)
    processor.process_queue.destroy()
    exit(0)