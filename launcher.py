import yaml
import logging

from pipeline import Pipe


def main():
    logging.basicConfig(filename='pipeline.log',
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)-8s %(message)s'
                        )
    logging.info('--- Pipeline START ---')

    with open('config.yaml')as f:
        config = yaml.safe_load(f)

    pipe = Pipe(config=config)
    pipe.execute()

    logging.info('--- Pipeline FINISH ---')


if __name__ == '__main__':
    main()
