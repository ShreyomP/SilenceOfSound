import io
import time
import base64
import subprocess
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import librosa
from datetime import datetime, timezone

from .config import FFMPEG_BINARY
from .spectrogram import generate_mel_spectrogram

# Module-level cache to remember active stream session folders across requests
_ACTIVE_FOLDER_CACHE = {}

class QuiltS3Fetcher:
    def __init__(self, node="rpi_port_townsend"):
        self.node = node
        self.bucket_name = "audio-orcasound-net"
        self.prefix = f"{node}/hls/"
        self.s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        self.playlist_url = None

    def _discover_latest_folder(self):
        """Scans S3 to discover the newest HLS session folder timestamp."""
        # Look back 24 hours, and fall back to 7 days if node has a long-running stream session
        lookback = int(time.time()) - 24 * 3600
        start_after = f"{self.prefix}{lookback}/"

        paginator = self.s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(
            Bucket=self.bucket_name,
            Prefix=self.prefix,
            Delimiter='/',
            StartAfter=start_after
        )

        all_prefixes = []
        for page in page_iterator:
            for cp in page.get('CommonPrefixes', []):
                all_prefixes.append(cp)

        # Long-running stream session fallback (up to 7 days)
        if not all_prefixes:
            lookback = int(time.time()) - 7 * 86400
            start_after = f"{self.prefix}{lookback}/"
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=self.prefix,
                Delimiter='/',
                StartAfter=start_after
            )
            for page in page_iterator:
                for cp in page.get('CommonPrefixes', []):
                    all_prefixes.append(cp)

        if not all_prefixes:
            return None

        all_prefixes.sort(key=lambda p: p['Prefix'])
        return all_prefixes[-1]['Prefix']

    def get_latest_spectrogram(self):
        try:
            folder = _ACTIVE_FOLDER_CACHE.get(self.node)

            # Fast-path: try fetching live.m3u8 from the known cached folder
            playlist_text = None
            playlist_key = None
            if folder:
                try:
                    playlist_key = f"{folder}live.m3u8"
                    obj = self.s3.get_object(Bucket=self.bucket_name, Key=playlist_key)
                    playlist_text = obj['Body'].read().decode('utf-8')
                except Exception:
                    folder = None  # Cache was invalid; fall back to discovery

            # Slow-path: discover new folder if not in cache or if fetch failed
            if not folder or not playlist_text:
                folder = self._discover_latest_folder()
                if not folder:
                    return None, None, None
                _ACTIVE_FOLDER_CACHE[self.node] = folder
                playlist_key = f"{folder}live.m3u8"
                playlist = self.s3.get_object(Bucket=self.bucket_name, Key=playlist_key)
                playlist_text = playlist['Body'].read().decode('utf-8')

            self.playlist_url = f"https://{self.bucket_name}.s3.amazonaws.com/{playlist_key}"

            segments = [line.strip() for line in playlist_text.split('\n') if line.strip().endswith('.ts')]
            if not segments:
                return None, None, None

            ts_key = f"{folder}{segments[-1]}"

            head = self.s3.head_object(Bucket=self.bucket_name, Key=ts_key)
            if head.get('ContentLength', 0) < 1000:
                return None, None, None

            # Skip segments older than 15 minutes; if stale, invalidate cached folder
            modified = head.get('LastModified')
            if modified:
                age = (datetime.now(timezone.utc) - modified).total_seconds()
                if age > 15 * 60:
                    _ACTIVE_FOLDER_CACHE.pop(self.node, None)
                    return None, None, None

            raw = self.s3.get_object(Bucket=self.bucket_name, Key=ts_key)['Body'].read()

            # Decode TS audio segment to mono wav using ffmpeg
            cmd = [
                FFMPEG_BINARY, '-i', 'pipe:0',
                '-f', 'wav', '-ar', '44100', '-ac', '1',
                'pipe:1', '-v', 'error'
            ]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, _ = proc.communicate(input=raw)

            y, sr = librosa.load(io.BytesIO(out), sr=44100, duration=2.0)

            # Generate standardized spectrogram without text or axis artifacts
            img, _ = generate_mel_spectrogram(y, sr=sr)

            # Base64 copy for the web client
            out_buf = io.BytesIO()
            img.save(out_buf, format="PNG")
            img_b64 = base64.b64encode(out_buf.getvalue()).decode("utf-8")

            return img, img_b64, head['LastModified']

        except Exception as e:
            print(f"Error fetching latest spectrogram for {self.node}: {e}")
            return None, None, None

