if __name__ == "__main__":

    # Load configuration file
    with open("config/config.json") as cfg:
        config = json.load(cfg)

    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("filepath", nargs=1,
                        help="Input image location", type=str)

    args = parser.parse_args()
