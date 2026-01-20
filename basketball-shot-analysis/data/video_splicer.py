"""
Video Downloader and Splicer for Basketball Shot Analysis
Downloads YouTube videos and extracts specific time segments for ML training.
"""

import os
import json
import csv
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import yt_dlp
from moviepy import VideoFileClip


class VideoDownloader:
    """Handles downloading videos from YouTube."""
    
    def __init__(self, output_dir: str = "raw_videos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, url: str, video_id: Optional[str] = None) -> str:
        """
        Download a YouTube video.
        
        Args:
            url: YouTube video URL
            video_id: Optional custom identifier for the video
            
        Returns:
            Path to downloaded video file
        """
        if video_id is None:
            # Extract video ID from URL
            video_id = url.split("watch?v=")[-1].split("&")[0]
        
        output_path = self.output_dir / f"{video_id}.mp4"
        
        # Skip if already downloaded
        if output_path.exists():
            print(f"Video {video_id} already exists, skipping download...")
            return str(output_path)
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': str(self.output_dir / f"{video_id}.%(ext)s"),
            'quiet': False,
            'no_warnings': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                print(f"Downloading: {url}")
                ydl.download([url])
                
                # Find the downloaded file
                downloaded_files = list(self.output_dir.glob(f"{video_id}.*"))
                if downloaded_files:
                    # Rename to .mp4 if needed
                    downloaded_file = downloaded_files[0]
                    if downloaded_file.suffix != '.mp4':
                        mp4_path = downloaded_file.with_suffix('.mp4')
                        subprocess.run(['ffmpeg', '-i', str(downloaded_file), 
                                      '-codec', 'copy', str(mp4_path), '-y'],
                                     check=True, capture_output=True)
                        downloaded_file.unlink()
                        return str(mp4_path)
                    return str(downloaded_file)
                else:
                    # Fallback: try mp4
                    if output_path.exists():
                        return str(output_path)
                    raise FileNotFoundError(f"Downloaded file not found for {video_id}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            raise


class VideoSplicer:
    """Handles cutting videos into segments."""
    
    def __init__(self, output_dir: str = "clips"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def splice(self, video_path: str, start_time: float, end_time: float, 
               output_name: str, fps: Optional[int] = None) -> str:
        """
        Extract a segment from a video.
        
        Args:
            video_path: Path to source video
            start_time: Start time in seconds
            end_time: End time in seconds
            output_name: Name for output clip (without extension)
            fps: Optional target FPS (default: keep original)
            
        Returns:
            Path to output clip
        """
        output_path = self.output_dir / f"{output_name}.mp4"
        
        try:
            clip = VideoFileClip(video_path).subclip(start_time, end_time)
            
            # Optionally change FPS
            if fps:
                clip = clip.set_fps(fps)
            
            clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                preset='medium',
                logger=None  # Suppress verbose output
            )
            clip.close()
            
            print(f"Created clip: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"Error creating clip {output_name}: {e}")
            raise


class BasketballVideoProcessor:
    """Main class for processing basketball shooting videos."""
    
    def __init__(self, raw_videos_dir: str = "raw_videos", clips_dir: str = "clips"):
        self.downloader = VideoDownloader(raw_videos_dir)
        self.splicer = VideoSplicer(clips_dir)
    
    def process_from_json(self, config_path: str):
        """
        Process videos from a JSON configuration file.
        
        JSON format:
        {
            "videos": [
                {
                    "url": "https://youtube.com/watch?v=...",
                    "video_id": "optional_custom_id",
                    "clips": [
                        {
                            "name": "shot_1",
                            "start": 10.5,
                            "end": 15.0
                        }
                    ]
                }
            ]
        }
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        for video_config in config['videos']:
            url = video_config['url']
            video_id = video_config.get('video_id')
            clips = video_config.get('clips', [])
            
            # Download video
            try:
                video_path = self.downloader.download(url, video_id)
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                continue
            
            # Extract clips
            for clip_config in clips:
                clip_name = clip_config['name']
                start = clip_config['start']
                end = clip_config['end']
                
                try:
                    self.splicer.splice(video_path, start, end, clip_name)
                except Exception as e:
                    print(f"Failed to create clip {clip_name}: {e}")
    
    def process_from_csv(self, config_path: str):
        """
        Process videos from a CSV configuration file.
        
        CSV format:
        url,video_id,clip_name,start_time,end_time
        https://youtube.com/watch?v=...,vid1,shot_1,10.5,15.0
        https://youtube.com/watch?v=...,vid1,shot_2,20.0,24.5
        """
        # Group by video to avoid multiple downloads
        video_map = {}
        
        with open(config_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row['url']
                video_id = row.get('video_id', '')
                clip_name = row['clip_name']
                start = float(row['start_time'])
                end = float(row['end_time'])
                
                # Group clips by video
                key = (url, video_id) if video_id else (url, None)
                if key not in video_map:
                    video_map[key] = []
                video_map[key].append((clip_name, start, end))
        
        # Process each video
        for (url, video_id), clips in video_map.items():
            # Download video
            try:
                video_path = self.downloader.download(url, video_id if video_id else None)
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                continue
            
            # Extract clips
            for clip_name, start, end in clips:
                try:
                    self.splicer.splice(video_path, start, end, clip_name)
                except Exception as e:
                    print(f"Failed to create clip {clip_name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Download YouTube videos and extract segments for ML training'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to configuration file (JSON or CSV)'
    )
    parser.add_argument(
        '--raw-dir',
        type=str,
        default='raw_videos',
        help='Directory for downloaded raw videos (default: raw_videos)'
    )
    parser.add_argument(
        '--clips-dir',
        type=str,
        default='clips',
        help='Directory for output clips (default: clips)'
    )
    
    args = parser.parse_args()
    
    processor = BasketballVideoProcessor(args.raw_dir, args.clips_dir)
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return
    
    if config_path.suffix == '.json':
        processor.process_from_json(str(config_path))
    elif config_path.suffix == '.csv':
        processor.process_from_csv(str(config_path))
    else:
        print(f"Error: Unsupported configuration format. Use .json or .csv")
        return
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()

