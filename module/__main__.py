from .processor import Processor


if(__name__ == '__main__'):
    processor = Processor()
    processor.process_from_file('data/crops/crop_3_3_0000.tif')
    processor.process_queue.thread.join()