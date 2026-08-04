from __future__ import annotations

import csv
import json
import struct
import zipfile
from pathlib import Path
from app.storage import publish_files


def _vlq(value:int)->bytes:
    buffer=value&0x7F; output=bytearray([buffer])
    while value>>7:
        value>>=7; output.insert(0,(value&0x7F)|0x80)
    return bytes(output)

def _marker_midi(markers:list[dict],path:Path,bpm:float=120.0)->Path:
    ticks_per_quarter=480; microseconds=int(round(60_000_000/bpm)); events=bytearray(); events.extend(b"\x00\xFF\x51\x03"+microseconds.to_bytes(3,"big")); previous_tick=0
    for marker in sorted(markers,key=lambda item:float(item["time_seconds"])):
        tick=int(round(float(marker["time_seconds"])*bpm/60.0*ticks_per_quarter)); delta=max(0,tick-previous_tick); previous_tick=tick; text=str(marker["name"]).encode("utf-8")[:255]; events.extend(_vlq(delta)); events.extend(b"\xFF\x06"); events.extend(_vlq(len(text))); events.extend(text)
    events.extend(b"\x00\xFF\x2F\x00"); path.write_bytes(b"MThd"+struct.pack(">IHHH",6,0,1,ticks_per_quarter)+b"MTrk"+struct.pack(">I",len(events))+bytes(events)); return path

def _markers_from_payload(payload:dict)->list[dict]:
    markers=[]
    for item in payload.get("markers") or []: markers.append({"time_seconds":float(item.get("time_seconds") or item.get("start") or 0.0),"name":str(item.get("name") or item.get("text") or "Marker")})
    for item in payload.get("sections") or []: markers.append({"time_seconds":float(item.get("time_seconds") or item.get("start") or 0.0),"name":str(item.get("name") or "Section")})
    for item in payload.get("lines") or []: markers.append({"time_seconds":float(item.get("start") or 0.0),"name":f"LYRIC: {item.get('text') or ''}"})
    markers.sort(key=lambda item:item["time_seconds"]); return markers

def _reaper_project(payload:dict,markers:list[dict])->str:
    lines=['<REAPER_PROJECT 0.1 "7.0" 1700000000',f"  SAMPLERATE {int(payload.get('sample_rate',48000))} 0 0","  TEMPO 120 4 4"]
    for index,marker in enumerate(markers,start=1): lines.append(f'  MARKER {index} {float(marker["time_seconds"]):.9f} "{str(marker["name"]).replace(chr(34),chr(39))}" 0 0 1 B')
    for index,stem in enumerate(payload.get("stems") or [],start=1):
        if isinstance(stem,str): name=f"Stem {index}"; filename=Path(stem).name
        else: name=str(stem.get("name") or f"Stem {index}"); filename=str(stem.get("filename") or Path(str(stem.get("url") or "stem.wav")).name)
        safe=name.replace(chr(34),chr(39)); file_safe=filename.replace(chr(34),chr(39)); lines.extend(["  <TRACK",f'    NAME "{safe}"',"    VOLPAN 1 0 -1 -1 1","    <ITEM","      POSITION 0","      LENGTH 3600",f'      NAME "{safe}"',"      <SOURCE WAVE",f'        FILE "{file_safe}"',"      >","    >","  >"])
    lines.append(">"); return "\n".join(lines)+"\n"

def export_daw_job(payload:dict)->dict:
    artist=str(payload.get("artist") or "sounddecay"); song=str(payload.get("song") or "untitled"); job_dir=Path("/tmp/stemforge_daw_v2"); job_dir.mkdir(parents=True,exist_ok=True); markers=_markers_from_payload(payload); marker_csv=job_dir/"markers.csv"
    with marker_csv.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle); writer.writerow(["time_seconds","timecode","name"])
        for marker in markers:
            seconds=float(marker["time_seconds"]); minutes,remainder=divmod(seconds,60); writer.writerow([f"{seconds:.6f}",f"{int(minutes):02d}:{remainder:06.3f}",marker["name"]])
    chapters=job_dir/"chapters.txt"; chapters.write_text("\n".join(f"{int(item['time_seconds']//60):02d}:{item['time_seconds']%60:05.2f} {item['name']}" for item in markers)+"\n",encoding="utf-8"); rpp=job_dir/f"{song.replace(' ','_')}.rpp"; rpp.write_text(_reaper_project(payload,markers),encoding="utf-8"); midi=_marker_midi(markers,job_dir/"markers.mid",bpm=float(payload.get("bpm",120.0))); manifest={"status":"completed","engine":"StemForge DAW interchange v2","artist":artist,"song":song,"marker_count":len(markers),"formats":["Reaper RPP","marker CSV","standard MIDI markers","chapter text"],"note":"The Reaper project references stem filenames. Put the referenced audio files in the same folder or relink them when opening the project."}; manifest_path=job_dir/"manifest.json"; manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8"); archive=job_dir/f"{song.replace(' ','_')}_DAW_Package.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as zf:
        for path in [marker_csv,chapters,rpp,midi,manifest_path]: zf.write(path,arcname=path.name)
    manifest["outputs"]=publish_files([archive,marker_csv,chapters,rpp,midi,manifest_path],artist=artist,category="daw_exports",ttl_seconds=int(payload.get("output_ttl_seconds",86400))); return manifest
