# Basketball Video Downloader and Splicer

This tool downloads YouTube videos and extracts specific time segments for training machine learning models on basketball shooting form analysis.

## Prerequisites

1. **FFmpeg**: Must be installed on your system
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

2. **Python Dependencies**: Install via pip
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Using JSON Configuration

1. Create a JSON configuration file (see `video_config.json` for example):

```json
{
    "videos": [
        {
            "url": "https://www.youtube.com/watch?v=VIDEO_ID",
            "video_id": "custom_id_optional",
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
```

2. Run the script:
```bash
python video_splicer.py video_config.json
```

### Using CSV Configuration

1. Create a CSV configuration file (see `video_config.csv` for example):

```csv
url,video_id,clip_name,start_time,end_time,description
https://www.youtube.com/watch?v=VIDEO_ID,vid1,shot_1,10.5,15.0,Free throw
```

2. Run the script:
```bash
python video_splicer.py video_config.csv
```

### Command Line Options

- `config`: Path to configuration file (required)
- `--raw-dir`: Directory for downloaded raw videos (default: `raw_videos`)
- `--clips-dir`: Directory for output clips (default: `clips`)

Example with custom directories:
```bash
python video_splicer.py video_config.json --raw-dir downloads --clips-dir training_data
```

## Output Structure

```
basketball-shot-analysis/
├── data/
│   ├── video_splicer.py
│   ├── video_config.json
│   └── video_config.csv
├── raw_videos/          # Full downloaded videos
│   ├── video_id_1.mp4
│   └── video_id_2.mp4
└── clips/               # Extracted segments
    ├── shot_form_1.mp4
    ├── shot_form_2.mp4
    └── shot_form_3.mp4
```

## Configuration File Formats

### JSON Format

- `url`: YouTube video URL (required)
- `video_id`: Custom identifier (optional, will extract from URL if not provided)
- `clips`: Array of clip specifications
  - `name`: Output filename (without extension)
  - `start`: Start time in seconds (required)
  - `end`: End time in seconds (required)
  - `description`: Optional description (for your records)

### CSV Format

Columns:
- `url`: YouTube video URL
- `video_id`: Custom identifier (optional, can be empty)
- `clip_name`: Output filename (without extension)
- `start_time`: Start time in seconds
- `end_time`: End time in seconds
- `description`: Optional description

**Note**: Multiple clips from the same video should share the same `url` and `video_id` to avoid re-downloading.

## Tips

1. **Finding Time Segments**: Use YouTube's playback controls to identify exact start/end times for shots
2. **Batch Processing**: Add multiple videos and clips to your configuration file for batch processing
3. **Avoid Re-downloads**: The script skips videos that are already downloaded in the raw_videos directory
4. **Video Quality**: Videos are downloaded in the best available quality (MP4 format preferred)

## Troubleshooting

- **FFmpeg not found**: Ensure FFmpeg is installed and available in your PATH
- **Download failures**: Check your internet connection and verify YouTube URLs are correct
- **Memory issues**: For very long videos, consider downloading and processing separately
- **Format issues**: If a video format is unsupported, the script will try to convert it to MP4

