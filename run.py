from simclr import SimCLR
import yaml


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    simclr = SimCLR(config)
    simclr.train()


if __name__ == "__main__":
    main()
