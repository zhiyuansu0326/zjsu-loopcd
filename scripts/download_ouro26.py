#!/usr/bin/env python3
"""Reuse the verified downloader for the second official Ouro base checkpoint."""

import download_model as downloader

downloader.REPO_ID = "ByteDance/Ouro-2.6B"
downloader.REVISION = "1ed04250da1a9936042725d302e81c8fa2ab5abd"
downloader.MODEL_DIR = downloader.ROOT / "models/Ouro-2.6B"
downloader.MANIFEST_PATH = downloader.ROOT / "models/ouro26_manifest.json"

if __name__ == "__main__":
    downloader.main()
