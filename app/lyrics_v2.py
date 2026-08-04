from __future__ import annotations

import json
import re
from pathlib import Path
from app.export import build_lyric_exports
from app.lyric_align import align_lyrics_job
from app.storage import publish_files

SECTION_PATTERN=re.compile(r"^\s*\[([^\]]+)\]\s*$")

def preprocess_lyrics(lyrics:str)->dict:
    cleaned=[]; sections=[]; current_section=None
    for raw in str(lyrics).splitlines():
        line=raw.strip()
        if not line: continue
        match=SECTION_PATTERN.match(line)
        if match:
            current_section=match.group(1).strip(); sections.append({"name":current_section,"first_clean_line_index":len(cleaned)+1}); continue
        cleaned.append(line)
    return {"lyrics":"\n".join(cleaned),"lines":cleaned,"sections":sections}

def _apply_display_rules(lines:list[dict],payload:dict)->tuple[list[dict],dict]:
    selected=list(lines); excluded=[]; skip_lines=max(0,int(payload.get("skip_intro_lines",0)))
    if skip_lines: excluded.extend([line.get("index") for line in selected[:skip_lines]]); selected=selected[skip_lines:]
    first_text=str(payload.get("first_display_text") or "").strip().lower()
    if first_text:
        matched_index=next((index for index,line in enumerate(selected) if first_text in str(line.get("text") or "").lower()),None)
        if matched_index is None: raise ValueError(f"first_display_text was not found: {first_text}")
        excluded.extend([line.get("index") for line in selected[:matched_index]]); selected=selected[matched_index:]
    master_duration=payload.get("master_duration_seconds")
    if master_duration is not None:
        master_duration=float(master_duration); retained=[]
        for line in selected:
            if float(line["start"])>=master_duration: excluded.append(line.get("index")); continue
            clipped={**line,"end":min(float(line["end"]),master_duration)}
            if clipped["end"]-float(clipped["start"])>=0.08: retained.append(clipped)
            else: excluded.append(line.get("index"))
        selected=retained
    maximum_duration=float(payload.get("maximum_caption_seconds",12.0)); minimum_duration=float(payload.get("minimum_caption_seconds",0.35)); next_gap=float(payload.get("maximum_hold_into_gap_seconds",0.2)); adjusted=[]
    for index,line in enumerate(selected):
        item={**line}; start=float(item["start"]); end=min(float(item["end"]),start+maximum_duration)
        if index+1<len(selected): end=min(end,float(selected[index+1]["start"])-max(0.0,next_gap))
        item["end"]=max(start+minimum_duration,end); adjusted.append(item)
    for new_index,item in enumerate(adjusted,start=1): item["source_alignment_index"]=item.get("index"); item["index"]=new_index
    return adjusted,{"excluded_source_indices":[value for value in excluded if value is not None],"skip_intro_lines":skip_lines,"first_display_text":first_text or None,"master_duration_seconds":master_duration,"maximum_caption_seconds":maximum_duration}

def align_lyrics_smart_job(payload:dict)->dict:
    original_lyrics=str(payload.get("lyrics") or "")
    if not original_lyrics.strip(): return {"status":"rejected","reason":"lyrics is required."}
    prepared=preprocess_lyrics(original_lyrics); result=align_lyrics_job({**payload,"lyrics":prepared["lyrics"]})
    if result.get("status")!="completed": return result
    filtered,display_rules=_apply_display_rules(result.get("lines") or [],payload); output_dir=Path("/runpod-volume/stemforge/output")/"smart_lyrics"; exports=build_lyric_exports(str(payload.get("artist") or "sounddecay"),str(payload.get("song") or "untitled"),filtered,output_dir); manifest={"status":"completed","engine":"StemForge smart lyric pipeline v2","aligner_version":result.get("aligner_version"),"device":result.get("device"),"model":result.get("model"),"language":result.get("language"),"sections":prepared["sections"],"display_rules":display_rules,"raw_metrics":result.get("metrics"),"raw_review_line_indices":result.get("review_line_indices"),"line_count":len(filtered),"lines":filtered}; manifest_path=output_dir/"smart_alignment_manifest.json"; manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8"); paths=[Path(path) for path in exports.values()]+[manifest_path]; manifest["outputs"]=publish_files(paths,artist=str(payload.get("artist") or "sounddecay"),category="lyric_timings",ttl_seconds=int(payload.get("output_ttl_seconds",86400))); return manifest
