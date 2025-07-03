from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

from icecream import ic
from natsort import natsorted
from simplecv.video_utils import reencode_video_optimal


@dataclass
class EncodeConfig:
    input_dir: Path
    output_dir: Path


def reencode_videos(config: EncodeConfig):
    start_time: float = timer()
    end_time: float = timer()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Re-encoding videos from {config.input_dir} to {config.output_dir}")
    # Here you would implement the actual video re-encoding logic
    sequence_paths: list[Path] = natsorted([d for d in config.input_dir.glob("*") if d.is_dir()])
    # should be a total of 362 sequences, more info here https://github.com/assembly-101/assembly101-download-scripts
    ic(len(sequence_paths), "sequences found")
    bad_sequences: list[Path] = []
    for sequence_path in sequence_paths[:1]:
        exo_videos: list[Path] = natsorted(sequence_path.glob("C*.mp4"))
        ego_videos: list[Path] = natsorted(sequence_path.glob("HMC_*.mp4"))
        if len(exo_videos) != 8 or len(ego_videos) != 4:
            bad_sequences.append(sequence_path)
            continue
        for exo_video in exo_videos:
            reencode_video_optimal(
                input_video_path=exo_video,
                resize="720p",
                save_file=True,
                output_directory=config.output_dir / sequence_path.name,
            )
        for ego_video in ego_videos:
            reencode_video_optimal(
                input_video_path=ego_video,
                save_file=True,
                output_directory=config.output_dir / sequence_path.name,
            )
    ic(bad_sequences)
    ic(len(bad_sequences), "bad sequences found")
    # For now, we just simulate the process with a print statement
    print(f"Re-encoded videos in {end_time - start_time:.2f} seconds.")
