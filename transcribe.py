from faster_whisper import WhisperModel
from pathlib import Path
import srt, datetime, os

VIDEOS = Path("videos")
OUT = Path("captions")
OUT.mkdir(exist_ok=True)

LANGUAGE = "en"
MODEL_SIZE = os.getenv("WHISPER_MODEL", "medium")

model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
COMMON_KW = dict(language=LANGUAGE, task="transcribe", vad_filter=True, beam_size=5)

def to_srt(infile: Path) -> Path:
    segments, _ = model.transcribe(str(infile), **COMMON_KW)
    subs=[]
    for i, seg in enumerate(segments, start=1):
        subs.append(srt.Subtitle(
            index=i,
            start=datetime.timedelta(seconds=seg.start),
            end=datetime.timedelta(seconds=seg.end),
            content=seg.text.strip()))
    srt_path = OUT / (infile.stem + ".srt")
    srt_path.write_text(srt.compose(subs), encoding="utf-8")
    print(f"wrote {srt_path}")
    return srt_path

def srt_to_vtt(srt_path: Path) -> Path:
    text = srt_path.read_text(encoding="utf-8")
    vtt = "WEBVTT\n\n" + text.replace(",", ".")
    vtt_path = srt_path.with_suffix(".vtt")
    vtt_path.write_text(vtt, encoding="utf-8")
    print(f"wrote {vtt_path}")
    return vtt_path

if __name__ == "__main__":
    for f in sorted(VIDEOS.glob("*")):
        if f.suffix.lower() in {".mp4", ".mov", ".mkv", ".m4v"}:
            srt_to_vtt(to_srt(f))
