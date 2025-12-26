#!/usr/bin/env python3
import argparse
import os
import platform
import sys
from datetime import datetime


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="Jihun", help="your name")
    p.add_argument("--write", action="store_true", help="write a small output file")
    args = p.parse_args()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=== GitHub Python Test ===")
    print(f"Name      : {args.name}")
    print(f"Time      : {now}")
    print(f"Python    : {sys.version.split()[0]}")
    print(f"Platform  : {platform.platform()}")
    print(f"CWD       : {os.getcwd()}")

    if args.write:
        out = "github_test_output.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write("Hello GitHub!\n")
            f.write(f"Name: {args.name}\n")
            f.write(f"Time: {now}\n")
        print(f"Written   : {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
