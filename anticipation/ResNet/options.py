import argparse

str2bool = lambda arg: arg.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description="Train model for video-based surgical instrument anticipation."
)
parser.register("type", "bool", str2bool)

# input data
parser.add_argument("--width", type=int, default=384)
parser.add_argument("--height", type=int, default=216)
parser.add_argument("--data_folder", type=str, default="../data/images/")
parser.add_argument(
    "--annotation_folder", type=str, default="../data/annotations/tool_annotations/"
)
parser.add_argument(
    "--output_folder",
    type=str,
    default="/home/shaon/Desktop/Team project/implementation/ins_ant_resnet/output/experiments/",
)
parser.add_argument("--trial_name", type=str, default="anticipation")
# model
parser.add_argument("--num_class", type=int, default=3)
parser.add_argument("--num_ins", type=int, default=5)
parser.add_argument("--drop_prob", type=float, default=0.2)
parser.add_argument("--horizon", type=int, default=5)
# training
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--loss_scale", type=float, default=1e-2)
# evaluation
parser.add_argument("--model_save_freq", type=int, default=100)
parser.add_argument("--sample_freq", type=int, default=100)
parser.add_argument("--num_samples", type=int, default=10)
# testing
parser.add_argument("--test_folder", type=str, default="../output/test/")
parser.add_argument(
    "--model_folder",
    type=str,
    default="../output/experiments/20200821-1203_horizon5_anticipationTest/models/",
)
parser.add_argument("--model_epoch", type=int, default=100)
