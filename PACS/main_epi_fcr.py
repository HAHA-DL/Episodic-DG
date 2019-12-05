import argparse

from model_resnet import ModelEpiFCR


def main():
    train_arg_parser = argparse.ArgumentParser(description="parser")
    train_arg_parser.add_argument("--seed", type=int, default=1,
                                  help="")
    train_arg_parser.add_argument("--test_every", type=int, default=50,
                                  help="")
    train_arg_parser.add_argument("--batch_size", type=int, default=20,
                                  help="")
    train_arg_parser.add_argument("--num_classes", type=int, default=10,
                                  help="")
    train_arg_parser.add_argument("--step_size", type=int, default=10000,
                                  help="")
    train_arg_parser.add_argument("--loops_train", type=int, default=200000,
                                  help="")
    train_arg_parser.add_argument("--loops_warm", type=int, default=200000,
                                  help="")
    train_arg_parser.add_argument("--loops_agg_warm", type=int, default=200000,
                                  help="")
    train_arg_parser.add_argument("--unseen_index", type=int, default=0,
                                  help="")
    train_arg_parser.add_argument("--ite_train_epi_c", type=int, default=200000,
                                  help="")
    train_arg_parser.add_argument("--ite_train_epi_f", type=int, default=200000,
                                  help="")
    train_arg_parser.add_argument("--bn_eval", type=int, default=1,
                                  help="")
    train_arg_parser.add_argument("--warm_up_agg", type=int, default=-111,
                                  help='')
    train_arg_parser.add_argument("--lr", type=float, default=0.0001,
                                  help='')
    train_arg_parser.add_argument("--weight_decay", type=float, default=0.00005,
                                  help='')
    train_arg_parser.add_argument("--momentum", type=float, default=0.9,
                                  help='')
    train_arg_parser.add_argument("--loss_weight_epif", type=float, default=0.8,
                                  help='')
    train_arg_parser.add_argument("--loss_weight_epic", type=float, default=0.8,
                                  help='')
    train_arg_parser.add_argument("--loss_weight_epir", type=float, default=0.8,
                                  help='')
    train_arg_parser.add_argument("--logs", type=str, default='logs/',
                                  help='')
    train_arg_parser.add_argument("--model_path", type=str, default='',
                                  help='')
    train_arg_parser.add_argument("--state_dict", type=str, default='',
                                  help='')
    train_arg_parser.add_argument("--data_root", type=str, default='',
                                  help='')
    train_arg_parser.add_argument("--deterministic", type=bool, default=False,
                                  help='')

    args = train_arg_parser.parse_args()
    model_obj = ModelEpiFCR(flags=args)
    model_obj.train(flags=args)


if __name__ == "__main__":
    main()
