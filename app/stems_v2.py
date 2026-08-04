from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy import signal
from app.analysis_v2 import analyze_array
from app.audio_core import ensure_pcm_wav, load_audio, write_audio
from app.storage import materialize_input, publish_files


def _stem_specifications(payload:dict)->list[dict]:
    stems=payload.get("stems") or payload.get("stem_urls") or []
    if isinstance(stems,dict): stems=[{"name":name,"url":url} for name,url in stems.items()]
    output=[]
    for index,item in enumerate(stems):
        if isinstance(item,str): output.append({"name":f"stem_{index+1}","url":item})
        elif isinstance(item,dict): output.append({"name":str(item.get("name") or f"stem_{index+1}"),**item})
    return output


def _materialize_stem(spec:dict,job_dir:Path,index:int)->tuple[str,Path]:
    name=str(spec.get("name") or f"stem_{index+1}"); proxy={}
    for suffix in ("url","storage_key","base64","path"):
        if spec.get(suffix) is not None: proxy[f"stem_{suffix}"]=spec[suffix]
    source=materialize_input(proxy,"stem",job_dir/f"stem_{index}_source"); return name,ensure_pcm_wav(source,job_dir/f"stem_{index}.wav")


def _onset_envelope(audio:np.ndarray,sample_rate:int,rate_hz:int=100)->np.ndarray:
    mono=np.mean(audio,axis=1); frame=max(1,sample_rate//rate_hz); energy=np.sqrt(np.add.reduceat(np.square(mono,dtype=np.float64),np.arange(0,len(mono),frame))/frame+1e-20); novelty=np.maximum(0.0,np.diff(energy,prepend=energy[0])); novelty-=np.mean(novelty); return novelty/(np.linalg.norm(novelty)+1e-20)


def _estimate_offset(stem:np.ndarray,master:np.ndarray,sample_rate:int,maximum_seconds:float=8.0)->dict:
    stem_env=_onset_envelope(stem,sample_rate); master_env=_onset_envelope(master,sample_rate); correlation=signal.correlate(master_env,stem_env,mode="full",method="fft"); lags=signal.correlation_lags(len(master_env),len(stem_env),mode="full"); mask=np.abs(lags)<=int(maximum_seconds*100); selected=correlation[mask]; selected_lags=lags[mask]
    if not len(selected): return {"offset_seconds":0.0,"confidence":0.0}
    best=int(np.argmax(selected)); lag=int(selected_lags[best]); score=float(selected[best]/(np.linalg.norm(selected)+1e-20)*np.sqrt(len(selected))); return {"offset_seconds":round(lag/100.0,3),"confidence":round(max(0.0,min(score,1.0)),4),"method":"onset-envelope cross-correlation"}


def _align_length(audio:np.ndarray,target_length:int,offset_samples:int=0)->np.ndarray:
    output=np.zeros((target_length,audio.shape[1]),dtype=np.float32); source_start=max(0,-offset_samples); destination_start=max(0,offset_samples); available=min(len(audio)-source_start,target_length-destination_start)
    if available>0: output[destination_start:destination_start+available]=audio[source_start:source_start+available]
    return output


def _fingerprint(audio:np.ndarray,sample_rate:int)->np.ndarray:
    mono=np.mean(audio,axis=1)
    if len(mono)>sample_rate*180: mono=mono[:sample_rate*180]
    freqs,times,spectrum=signal.spectrogram(mono,sample_rate,nperseg=2048,noverlap=1536,mode="magnitude"); bands=np.array_split(np.arange(len(freqs)),32); fingerprint=np.array([np.mean(np.log1p(spectrum[index])) for index in bands],dtype=np.float64); fingerprint-=fingerprint.mean(); return fingerprint/(np.linalg.norm(fingerprint)+1e-20)


def inspect_stems_v2_job(payload:dict)->dict:
    artist=str(payload.get("artist") or "sounddecay"); song=str(payload.get("song") or "untitled"); specs=_stem_specifications(payload)
    if not specs:return {"status":"rejected","reason":"stems must contain at least one stem."}
    job_dir=Path("/tmp/stemforge_stems_v2"); job_dir.mkdir(parents=True,exist_ok=True); master_path=materialize_input(payload,"master",job_dir/"master_source",required=False); master_audio=None; sample_rate=48000; master_metrics=None
    if master_path: master_pcm=ensure_pcm_wav(master_path,job_dir/"master.wav"); master_audio,sample_rate=load_audio(master_pcm); master_metrics=analyze_array(master_audio,sample_rate)
    loaded=[]
    for index,spec in enumerate(specs):
        name,path=_materialize_stem(spec,job_dir,index); audio,stem_rate=load_audio(path)
        if stem_rate!=sample_rate: raise RuntimeError("Internal stem sample-rate normalization failed.")
        offset=_estimate_offset(audio,master_audio,sample_rate) if master_audio is not None else {"offset_seconds":0.0,"confidence":0.0,"method":"no master supplied"}; loaded.append({"name":name,"path":path,"audio":audio,"metrics":analyze_array(audio,sample_rate),"offset":offset,"fingerprint":_fingerprint(audio,sample_rate)})
    duplicates=[]
    for left,right in combinations(loaded,2):
        similarity=float(np.dot(left["fingerprint"],right["fingerprint"]))
        if similarity>=0.985: duplicates.append({"stem_a":left["name"],"stem_b":right["name"],"fingerprint_similarity":round(similarity,5)})
    reconstruction_report=None; output_paths=[]
    if master_audio is not None:
        target_length=len(master_audio); aligned=[_align_length(item["audio"],target_length,int(round(item["offset"]["offset_seconds"]*sample_rate))) for item in loaded]; reconstruction=np.sum(aligned,axis=0,dtype=np.float64).astype(np.float32); denominator=float(np.sum(np.square(reconstruction,dtype=np.float64))+1e-20); gain=float(np.sum(master_audio*reconstruction,dtype=np.float64)/denominator); reconstruction*=gain; residual=master_audio-reconstruction; residual_ratio_db=10.0*np.log10((float(np.mean(np.square(residual,dtype=np.float64)))+1e-20)/(float(np.mean(np.square(master_audio,dtype=np.float64)))+1e-20)); reconstruction_report={"least_squares_gain":round(gain,6),"null_residual_relative_db":round(residual_ratio_db,3),"quality":"strong" if residual_ratio_db<=-20 else "partial" if residual_ratio_db<=-10 else "poor","safe_for_stem_remix":residual_ratio_db<=-12}; reconstruction_path=job_dir/f"{song.replace(' ','_')}_stem_sum.wav"; residual_path=job_dir/f"{song.replace(' ','_')}_null_residual.wav"; write_audio(reconstruction_path,reconstruction,sample_rate); write_audio(residual_path,residual,sample_rate); output_paths.extend([reconstruction_path,residual_path])
    report={"status":"completed","engine":"StemForge stem integrity and reconstruction engine v2","artist":artist,"song":song,"master_metrics":master_metrics,"stem_count":len(loaded),"stems":[{"name":item["name"],"metrics":item["metrics"],"estimated_timeline_offset":item["offset"],"silent":item["metrics"]["rms_dbfs"]<-90.0,"clipped":item["metrics"]["clipped_sample_count"]>0} for item in loaded],"possible_duplicates":duplicates,"reconstruction":reconstruction_report,"honesty_note":"Timeline offsets estimated from onset envelopes can be unreliable for sparse, ambient, or effect-only stems. Low-confidence offsets should be checked manually."}; report_path=job_dir/"stem_integrity_report.json"; report_path.write_text(json.dumps(report,indent=2),encoding="utf-8"); output_paths.append(report_path); report["outputs"]=publish_files(output_paths,artist=artist,category="stem_reports",ttl_seconds=int(payload.get("output_ttl_seconds",86400))); return report
