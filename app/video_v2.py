from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from app.export import build_lyric_exports
from app.storage import materialize_input, publish_files

FORMATS={"vertical":(1080,1920),"horizontal":(1920,1080),"square":(1080,1080)}

def _run(command:list[str],timeout:int=14400)->None: subprocess.run(command,check=True,capture_output=True,text=True,timeout=timeout)

def _materialize_subtitles(payload:dict,job_dir:Path)->Path:
    existing=materialize_input(payload,"subtitles",job_dir/"lyrics.ass",required=False)
    if existing:
        if existing.suffix.lower() not in {".ass",".ssa",".srt"}: target=job_dir/"lyrics.ass"; shutil.copy2(existing,target); return target
        return existing
    lines=payload.get("lines") or []
    if not lines: raise ValueError("subtitles input or lines is required.")
    return Path(build_lyric_exports(str(payload.get("artist") or "sounddecay"),str(payload.get("song") or "untitled"),lines,job_dir)["ass_path"])

def _is_image(path:Path)->bool:return path.suffix.lower() in {".png",".jpg",".jpeg",".webp",".bmp"}
def _escape_filter_path(path:Path)->str:return str(path).replace("\\","\\\\").replace(":","\\:").replace("'","\\'")

def render_lyric_video_job(payload:dict)->dict:
    artist=str(payload.get("artist") or "sounddecay"); song=str(payload.get("song") or "untitled"); requested=payload.get("formats") or ["vertical","horizontal","square"]
    if isinstance(requested,str):requested=[requested]
    invalid=[name for name in requested if name not in FORMATS]
    if invalid:return {"status":"rejected","reason":f"Unknown formats: {invalid}","available_formats":sorted(FORMATS)}
    job_dir=Path("/tmp/stemforge_video_v2")
    if job_dir.exists():shutil.rmtree(job_dir)
    job_dir.mkdir(parents=True,exist_ok=True); audio=materialize_input(payload,"audio",job_dir/"audio"); background=materialize_input(payload,"background",job_dir/"background"); logo=materialize_input(payload,"logo",job_dir/"logo.png",required=False); subtitles=_materialize_subtitles(payload,job_dir); grain=float(payload.get("grain_strength",3.0)); zoom_speed=float(payload.get("zoom_speed",0.00015)); outputs=[]; commands=[]
    for format_name in requested:
        width,height=FORMATS[format_name]; output_path=job_dir/f"{song.replace(' ','_')}_{format_name}_lyric_video.mp4"; fps=int(payload.get("fps",30)); base_filter=f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},format=yuv420p"
        if _is_image(background):base_filter=f"scale={int(width*1.08)}:{int(height*1.08)}:force_original_aspect_ratio=increase,crop={width}:{height},zoompan=z='min(zoom+{zoom_speed},1.06)':d=1:s={width}x{height}:fps={fps},format=yuv420p"
        subtitle_filter=f"subtitles='{_escape_filter_path(subtitles)}':fontsdir='/usr/share/fonts/truetype'"; filters=[base_filter,f"noise=alls={grain}:allf=t+u",subtitle_filter]; command=["ffmpeg","-hide_banner","-loglevel","error","-y"]
        if _is_image(background):command.extend(["-loop","1","-framerate",str(fps),"-i",str(background)])
        else:command.extend(["-stream_loop","-1","-i",str(background)])
        command.extend(["-i",str(audio)])
        if logo:
            command.extend(["-i",str(logo)]); filter_complex=f"[0:v]{','.join(filters[:-1])}[base];[2:v]scale={max(80,width//8)}:-1[logo];[base][logo]overlay=W-w-{max(24,width//40)}:H-h-{max(24,height//40)}[withlogo];[withlogo]{filters[-1]}[v]"; command.extend(["-filter_complex",filter_complex,"-map","[v]","-map","1:a:0"])
        else:command.extend(["-vf",",".join(filters),"-map","0:v:0","-map","1:a:0"])
        command.extend(["-c:v","libx264","-preset",str(payload.get("preset") or "medium"),"-crf",str(payload.get("crf",18)),"-c:a","aac","-b:a","320k","-shortest","-movflags","+faststart",str(output_path)]); _run(command,timeout=int(payload.get("timeout_seconds",14400))); outputs.append(output_path); commands.append({"format":format_name,"resolution":[width,height]})
    manifest={"status":"completed","engine":"StemForge deterministic FFmpeg lyric-video renderer v2","song":song,"formats":commands,"generative_video_used":False,"note":"The renderer composites real supplied media, deterministic motion, grain, logo, audio, and timed subtitles. It does not synthesize people or narrative footage."}; manifest_path=job_dir/"video_manifest.json"; manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8"); outputs.append(manifest_path); manifest["outputs"]=publish_files(outputs,artist=artist,category="lyric_videos",ttl_seconds=int(payload.get("output_ttl_seconds",86400))); return manifest
