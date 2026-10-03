"""Record a short gameplay video (MP4) of the browser demo for sharing.

usage: python tools/make_videos.py
Needs Google Chrome, Node.js and `pip install imageio-ffmpeg`. Output: demo/<name>.mp4
The demo pages are driven step by step, so each video shows the real simulated game.
"""
import glob
import os
import shutil
import subprocess
import tempfile
import time

import imageio_ffmpeg

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
# project, output name, captured frames, game steps per captured frame, playback fps, seed
JOBS = [("", "snake_demo", 420, 1, 14, None)]


def main():
    profile = tempfile.mkdtemp()
    chrome = subprocess.Popen([CHROME, "--headless=new", "--remote-debugging-port=9333", "--hide-scrollbars",
                               f"--user-data-dir={profile}", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        for project, name, n, steps, fps, _ in JOBS:
            frames = tempfile.mkdtemp()
            url = "file:///" + os.path.join(REPO, project, "demo", "index.html").replace("\\", "/")
            r = subprocess.run(["node", os.path.join(REPO, "tools", "record_demo.mjs"), url, frames, str(n), str(steps)],
                               capture_output=True, text=True)
            print(project, r.stdout.strip() or r.stderr.strip()[-300:])
            out = os.path.join(REPO, project, "demo", name + ".mp4")
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                            os.path.join(frames, "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "20", "-movflags", "+faststart", out], check=True)
            print("  ->", out, f"{os.path.getsize(out) / 1e6:.1f} MB,", len(glob.glob(os.path.join(frames, '*.png'))),
                  "frames,", f"{n / fps:.0f} s")
            shutil.rmtree(frames, ignore_errors=True)
    finally:
        chrome.terminate()
        time.sleep(1)
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
