import subprocess
import argparse
import datetime

def save_twitch_stream(channel_name: str):
    twitch_url = f"https://www.twitch.tv/{channel_name}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"twitch_stream_{timestamp}.mp4"

    # Streamlink command to get stream
    streamlink_command = [
        "streamlink", twitch_url, "best", "--stdout"
    ]

    ffmpeg_command = [
        "ffmpeg",
        "-i", "-",           # Use streamlink's stdout as input
        "-c", "copy",        # Copy codec without re-encoding
        output_file          # Output filename
    ]

    with subprocess.Popen(streamlink_command, stdout=subprocess.PIPE) as streamlink_proc:
        with subprocess.Popen(ffmpeg_command, stdin=streamlink_proc.stdout) as ffmpeg_proc:
            try:
                ffmpeg_proc.communicate()
            except KeyboardInterrupt:
                print("\nStream capture interrupted.")
            finally:
                streamlink_proc.terminate()
                ffmpeg_proc.terminate()

    print(f"Stream saved to {output_file}")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--source", type=str, help="twitch channel name source")
    opts = args.parse_args()
    save_twitch_stream(opts.source)

