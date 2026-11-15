"""Frame access for external games: a video file or a folder of images, one interface.

``iter_frames`` yields ``(frame_number, image)`` in order, where ``frame_number`` is
in the *annotation's* numbering (``Game.frame_offset`` + position), so callers can
join frames to :func:`training.manifest.load_annotations` without arithmetic.

Videos are decoded strictly sequentially — ``grab()`` for skipped frames, ``read()``
for kept ones — and never by seeking: ``cv2.CAP_PROP_POS_FRAMES`` is not
frame-accurate on every H.264 file, and a one-frame slip would silently shift every
ball label. cv2 is imported inside the functions, so importing this module needs
nothing beyond the standard library.
"""

from __future__ import annotations

from collections.abc import Iterator

from training.compliance import assert_external
from training.manifest import Game, list_frame_files

TRUNCATION_TOLERANCE = 5   # frames a container's count may be off by


def frame_size(game: Game) -> tuple[int, int]:
    """``(width, height)`` in pixels of the game's frames."""
    import cv2

    assert_external(game.source, what=f"frames of game {game.id}")
    if game.video is not None:
        cap = cv2.VideoCapture(str(game.video))
        if not cap.isOpened():
            raise IOError(f"cannot open video: {game.video}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return width, height
    files = list_frame_files(game.frames_dir)
    if not files:
        raise IOError(f"no images in {game.frames_dir}")
    image = cv2.imread(str(files[0]))
    if image is None:
        raise IOError(f"cannot read image: {files[0]}")
    return int(image.shape[1]), int(image.shape[0])


def frame_count(game: Game) -> int:
    """Number of frames (the container's count for a video; exact for a folder)."""
    import cv2

    assert_external(game.source, what=f"frames of game {game.id}")
    if game.video is not None:
        cap = cv2.VideoCapture(str(game.video))
        if not cap.isOpened():
            raise IOError(f"cannot open video: {game.video}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total
    return len(list_frame_files(game.frames_dir))


def iter_frames(game: Game, step: int = 1, start: int = 0,
                max_frames: int | None = None) -> Iterator[tuple[int, object]]:
    """Yield ``(frame_number, BGR image)`` every ``step`` frames, in order.

    ``start`` is a position (0-based index into the game's frames), not an
    annotation frame number; ``max_frames`` caps how many frames are yielded.
    """
    import cv2

    if step < 1:
        raise ValueError("step must be >= 1")
    if start < 0:
        raise ValueError("start must be >= 0")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    assert_external(game.source, what=f"frames of game {game.id}")
    yielded = 0

    if game.video is not None:
        cap = cv2.VideoCapture(str(game.video))
        if not cap.isOpened():
            raise IOError(f"cannot open video: {game.video}")
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            position = 0
            while True:
                keep = position >= start and (position - start) % step == 0
                if keep:
                    ok, image = cap.read()
                    if not ok:
                        break
                    yield game.frame_offset + position, image
                    yielded += 1
                    if max_frames is not None and yielded >= max_frames:
                        return
                elif not cap.grab():
                    break
                position += 1
        finally:
            cap.release()
        # Decoding ended before the container's frame count: a truncated or corrupt
        # file. Container counts can be off by a few frames, hence the tolerance.
        if expected > 0 and expected - position > TRUNCATION_TOLERANCE:
            raise IOError(f"{game.video}: decoding stopped at frame {position} of"
                          f" {expected} — truncated or corrupt video")
        return

    files = list_frame_files(game.frames_dir)
    shape = None
    for position in range(start, len(files), step):
        image = cv2.imread(str(files[position]))
        if image is None:
            raise IOError(f"cannot read image: {files[position]}")
        if shape is None:
            shape = image.shape[:2]
        elif image.shape[:2] != shape:
            raise IOError(f"{files[position]}: size {image.shape[1]}x{image.shape[0]}"
                          f" differs from the first frame ({shape[1]}x{shape[0]})")
        yield game.frame_offset + position, image
        yielded += 1
        if max_frames is not None and yielded >= max_frames:
            break
