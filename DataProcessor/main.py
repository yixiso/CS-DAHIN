import argparse
from pre_imdb_time import pre_imdb_time
from pre_dblp_time import pre_dblp_time
from pre_acm_time import pre_acm_time



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='?', default='IMDB',
                        help='Dataset name to processing.')
    parser.add_argument(
        "--use_cache",
        type=str,
        default="none",
        help="If use cache.(yes/no)"
    )
    parser.add_argument(
        "--time_scale",
        type=int, 
        default=10, 
        help="Time scale."
    )
    args = parser.parse_args().__dict__
    print(args)
    
    if args["dataset"] == "imdb_time":
        pre_imdb_time(args)
    elif args["dataset"] == "dblp_time":
        pre_dblp_time(args)
    elif args["dataset"] == "acm_time":
        pre_acm_time(args)