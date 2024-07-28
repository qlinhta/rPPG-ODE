import os
import subprocess
import sys


def convert_avi_to_mp4(input_file: str, output_file: str) -> None:
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"The input file {input_file} does not exist.")

    command = [
        'ffmpeg',
        '-i', input_file,
        '-vcodec', 'libx264',
        '-crf', '23',
        '-preset', 'medium',
        '-acodec', 'aac',
        '-strict', 'experimental',
        output_file
    ]

    subprocess.run(command, check=True)


def main(input_file: str, output_file: str) -> None:
    try:
        convert_avi_to_mp4(input_file, output_file)
        print(f"Successfully converted {input_file} to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py <input_file.avi> <output_file.mp4>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    main(input_file, output_file)
