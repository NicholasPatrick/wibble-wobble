import argparse
import cv2
from wobble import wobble


def main():
    parser = argparse.ArgumentParser(
        prog="wibble-wobble",
        description="exaggerates the movement in a video",
        epilog="github.com/NicholasPatrick/wibble-wobble",
    )
    parser.add_argument("filename")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.mp4",
        help="output file name, defaults to output.mp4",
    )
    parser.add_argument(
        "-f", "--factor", type=int, default=30, help="wobble factor, defaults to 30"
    )
    parser.add_argument(
        "-b", "--block", type=int, default=30, help="block size, defaults to 30"
    )
    parser.add_argument(
        "-O",
        "--overlaps",
        action="store_true",
        help="if set, makes it run 4 times slower for a slightly nicer result",
    )
    args = parser.parse_args()
    video = cv2.VideoCapture(args.filename)
    wobble(video, args.output, args.factor, args.block, args.overlaps)
    video.release()


if __name__ == "__main__":
    main()
