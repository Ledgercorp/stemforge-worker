from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from app.analysis_v2 import analyze_array
from app.audio_core import apply_mid_side_width, ensure_pcm_wav, highpass, limit_peak, linked_compressor, load_audio, normalize_loudness, write_audio
from app.memory import get_memory
from app.storage import materialize_input, publish_files

PROFILES={
"dynamic":{"target_lufs":-14.0,"threshold_db":-10.0,"ratio":1.35,"attack_ms":35.0,"release_ms":240.0,"width":0.99,"ceiling_dbfs":-1.0},
"balanced":{"target_lufs":-12.5,"threshold_db":-14.0,"ratio":1.8,"attack_ms":25.0,"release_ms":190.0,"width":0.98,"ceiling_dbfs":-1.0},
"dense":{"target_lufs":-10.8,"threshold_db":-17.0,"ratio":2.5,"attack_ms":18.0,"release_ms":150.0,"width":0.96,"ceiling_dbfs":-0.9}}

def _render_candidate(source:np.ndarray,sample_rate:int,settings:dict)->tuple[np.ndarray,dict]:
    processed=highpass(source,sample_rate,cutoff_hz=20.0)
    processed=apply_mid_side_width(processed,float(settings["width"]))
    processed=linked_compressor(processed,sample_rate,threshold_db=float(settings["threshold_db"]),ratio=float(settings["ratio"]),attack_ms=float(settings["attack_ms"]),release_ms=float(settings["release_ms"]))
    processed,loudness=normalize_loudness(processed,sample_rate,float(settings["target_lufs"]))
    processed,limiter=limit_peak(processed,ceiling_dbfs=float(settings["ceiling_dbfs"]))
    return processed.astype(np.float32),{**settings,**loudness,**limiter}

def _safety(source_metrics:dict,candidate_metrics:dict)->dict:
    source_crest=float(source_metrics["crest_factor_db"]); candidate_crest=float(candidate_metrics["crest_factor_db"]); source_corr=source_metrics.get("stereo_correlation"); candidate_corr=candidate_metrics.get("stereo_correlation"); reasons=[]
    if candidate_metrics["clipped_sample_count"]>0: reasons.append("clipped_samples")
    if float(candidate_metrics["true_peak_dbtp"])>-0.75: reasons.append("true_peak_margin")
    if source_crest-candidate_crest>5.5: reasons.append("excessive_crest_reduction")
    if source_corr is not None and candidate_corr is not None and float(candidate_corr)<min(-0.05,float(source_corr)-0.25): reasons.append("stereo_phase_risk")
    return {"accepted":not reasons,"reasons":reasons,"crest_change_db":round(candidate_crest-source_crest,3),"stereo_correlation_change":round(float(candidate_corr)-float(source_corr),5) if source_corr is not None and candidate_corr is not None else None}

def mastering_plan(payload:dict)->dict:
    artist=str(payload.get("artist") or "sounddecay"); memory=get_memory(artist)
    return {"status":"completed","mode":"executable_candidate_plan","artist":artist,"memory_used":memory,"candidate_profiles":PROFILES,"guardrails":["No vocal time warping.","Length and sample rate are preserved.","Each render is measured after processing.","Candidates with clipping, severe phase change, or excessive crest loss are rejected.","Processing is deterministic and reversible by retaining the source."]}

def master_audio_job(payload:dict)->dict:
    artist=str(payload.get("artist") or "sounddecay"); song=str(payload.get("song") or "untitled"); requested=payload.get("profiles") or ["dynamic","balanced","dense"]
    if isinstance(requested,str): requested=[requested]
    invalid=[name for name in requested if name not in PROFILES]
    if invalid: return {"status":"rejected","reason":f"Unknown profiles: {invalid}","available_profiles":sorted(PROFILES)}
    job_dir=Path("/tmp/stemforge_mastering_v2"); job_dir.mkdir(parents=True,exist_ok=True); source_path=materialize_input(payload,"audio",job_dir/"source"); pcm=ensure_pcm_wav(source_path,job_dir/"source.wav"); source,sample_rate=load_audio(pcm); source_metrics=analyze_array(source,sample_rate); memory=get_memory(artist); candidates=[]; published_paths=[]
    for name in requested:
        settings={**PROFILES[name]}; settings.update((payload.get("profile_overrides") or {}).get(name) or {}); rendered,chain=_render_candidate(source,sample_rate,settings); metrics=analyze_array(rendered,sample_rate); safety=_safety(source_metrics,metrics); path=job_dir/f"{song.replace(' ','_')}_{name}_master.wav"; write_audio(path,rendered,sample_rate); published_paths.append(path); candidates.append({"profile":name,"chain":chain,"metrics":metrics,"safety":safety,"local_path":str(path)})
    report={"status":"completed","engine":"StemForge deterministic mastering candidate engine v2","artist":artist,"song":song,"memory_used":memory,"source_metrics":source_metrics,"candidates":candidates,"approval_required":True,"honesty_note":"These are conservative DSP candidates, not a claim of human mastering judgment. Choose by level-matched listening after reviewing the safety checks."}; report_path=job_dir/"mastering_report.json"; report_path.write_text(json.dumps(report,indent=2),encoding="utf-8"); published_paths.append(report_path); published=publish_files(published_paths,artist=artist,category="masters",ttl_seconds=int(payload.get("output_ttl_seconds",86400))); by_filename={item["filename"]:item for item in published}
    for candidate in candidates: candidate["output"]=by_filename[Path(candidate["local_path"]).name]
    report["report_output"]=by_filename[report_path.name]; return report
