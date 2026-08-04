from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from app.storage import materialize_input, publish_files


def separate_stems_job(payload: dict) -> dict:
    artist=str(payload.get("artist") or "sounddecay"); song=str(payload.get("song") or "untitled"); model=str(payload.get("model") or "htdemucs"); two_stems=payload.get("two_stems"); job_dir=Path("/tmp/stemforge_separation_v2")
    if job_dir.exists(): shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True,exist_ok=True); source=materialize_input(payload,"audio",job_dir/"source_audio"); output_root=job_dir/"separated"; command=["python","-m","demucs.separate","-n",model,"-o",str(output_root)]
    if two_stems:
        if str(two_stems) not in {"vocals","drums","bass","other"}: return {"status":"rejected","reason":"two_stems must be vocals, drums, bass, or other."}
        command.extend(["--two-stems",str(two_stems)])
    command.append(str(source))
    try: proc=subprocess.run(command,check=True,capture_output=True,text=True,timeout=int(payload.get("timeout_seconds",7200)))
    except FileNotFoundError as exc: raise RuntimeError("Demucs is not installed in the worker image.") from exc
    except subprocess.CalledProcessError as exc: return {"status":"error","reason":"Demucs separation failed.","stderr":exc.stderr[-4000:]}
    files=sorted(output_root.rglob("*.wav"))
    if not files: return {"status":"error","reason":"Demucs completed without producing WAV stems.","stdout":proc.stdout[-2000:],"stderr":proc.stderr[-2000:]}
    return {"status":"completed","engine":"Demucs source separation through StemForge","model":model,"two_stems":two_stems,"song":song,"outputs":publish_files(files,artist=artist,category="separated_stems",ttl_seconds=int(payload.get("output_ttl_seconds",86400))),"honesty_note":"Separated stems are estimates, not original multitracks. Phase leakage and artifacts are expected, especially in dense distorted music."}
