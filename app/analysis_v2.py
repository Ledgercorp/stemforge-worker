from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import signal

from app.audio_core import db, ensure_pcm_wav, ffprobe, integrated_loudness, load_audio, peak, rms, stereo_correlation, true_peak
from app.storage import materialize_input, publish_files

BANDS = [("sub",20.0,80.0),("bass",80.0,250.0),("low_mid",250.0,800.0),("mid",800.0,2500.0),("presence",2500.0,5000.0),("high",5000.0,12000.0),("air",12000.0,20000.0)]


def _spectral_distribution(mono: np.ndarray, sample_rate: int) -> dict:
    if len(mono) > sample_rate * 600:
        step = max(1, len(mono) // (sample_rate * 600)); mono = mono[::step]; sample_rate = max(1, sample_rate // step)
    freqs, psd = signal.welch(mono, sample_rate, nperseg=min(16384, max(2048, len(mono))))
    total = float(np.trapezoid(psd, freqs) + 1e-20)
    output = {}
    for name, low, high in BANDS:
        mask = (freqs >= low) & (freqs < min(high, sample_rate / 2.0))
        output[name] = round(float(np.trapezoid(psd[mask], freqs[mask]) / total), 7) if np.any(mask) else 0.0
    return output


def _timeline(audio: np.ndarray, sample_rate: int, window_seconds: float = 10.0) -> list[dict]:
    window = max(1, int(window_seconds * sample_rate)); values = []
    for start in range(0, len(audio), window):
        chunk = audio[start:start+window]
        if not len(chunk): continue
        chunk_rms, chunk_peak = rms(chunk), peak(chunk); correlation = stereo_correlation(chunk)
        values.append({"start_seconds":round(start/sample_rate,3),"end_seconds":round(min(len(audio),start+window)/sample_rate,3),"rms_dbfs":round(db(chunk_rms),3),"peak_dbfs":round(db(chunk_peak),3),"crest_factor_db":round(db(chunk_peak)-db(chunk_rms),3),"stereo_correlation":round(correlation,5) if correlation is not None else None})
    return values


def _silence_regions(audio: np.ndarray, sample_rate: int, *, threshold_dbfs: float = -60.0, minimum_seconds: float = 0.4) -> list[dict]:
    mono = np.max(np.abs(audio), axis=1); frame = max(1, int(sample_rate*0.05)); envelope = np.maximum.reduceat(mono, np.arange(0,len(mono),frame)); silent = envelope < 10.0**(threshold_dbfs/20.0); minimum_frames=max(1,int(minimum_seconds/0.05)); output=[]; start=None
    for index,value in enumerate(silent):
        if value and start is None: start=index
        if (not value or index==len(silent)-1) and start is not None:
            end=index if not value else index+1
            if end-start>=minimum_frames: output.append({"start_seconds":round(start*0.05,3),"end_seconds":round(min(len(audio)/sample_rate,end*0.05),3)})
            start=None
    return output


def _transient_density(audio: np.ndarray, sample_rate: int) -> float:
    mono=np.mean(audio,axis=1); envelope=np.abs(signal.hilbert(mono)); smoothing=max(1,int(sample_rate*0.01)); smooth=np.convolve(envelope,np.ones(smoothing)/smoothing,mode="same"); novelty=np.maximum(0.0,np.diff(smooth,prepend=smooth[0])); median=np.median(novelty); threshold=median+6.0*np.median(np.abs(novelty-median)); peaks,_=signal.find_peaks(novelty,height=max(threshold,1e-7),distance=max(1,int(sample_rate*0.05))); return float(len(peaks)/max(len(audio)/sample_rate,1e-6))


def analyze_array(audio: np.ndarray, sample_rate: int) -> dict:
    maximum, root_mean_square, tp = peak(audio), rms(audio), true_peak(audio,oversample=4); correlation=stereo_correlation(audio); derivative=np.abs(np.diff(audio,axis=0)); discontinuity_threshold=max(0.5,float(np.median(derivative)*35.0)); lufs=integrated_loudness(audio,sample_rate)
    return {"duration_seconds":round(len(audio)/sample_rate,6),"sample_rate":sample_rate,"channels":int(audio.shape[1]),"sample_peak_dbfs":round(db(maximum),3),"true_peak_dbtp":round(db(tp),3),"rms_dbfs":round(db(root_mean_square),3),"crest_factor_db":round(db(maximum)-db(root_mean_square),3),"integrated_lufs":round(lufs,3) if lufs is not None and np.isfinite(lufs) else None,"stereo_correlation":round(correlation,6) if correlation is not None else None,"dc_offset":[round(float(v),8) for v in np.mean(audio,axis=0)],"clipped_sample_count":int(np.count_nonzero(np.abs(audio)>=0.999969)),"waveform_discontinuity_count":int(np.count_nonzero(np.max(derivative,axis=1)>discontinuity_threshold)),"transient_density_per_second":round(_transient_density(audio,sample_rate),5),"spectral_energy_distribution":_spectral_distribution(np.mean(audio,axis=1),sample_rate),"silence_regions":_silence_regions(audio,sample_rate),"timeline_10s":_timeline(audio,sample_rate)}


def _build_plots(audio: np.ndarray, sample_rate: int, output_dir: Path, stem: str) -> list[Path]:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True,exist_ok=True); mono=np.mean(audio,axis=1); times=np.arange(len(mono))/sample_rate; decimation=max(1,len(mono)//300000)
    waveform_path=output_dir/f"{stem}_waveform.png"; fig=plt.figure(figsize=(14,4)); plt.plot(times[::decimation],mono[::decimation],linewidth=0.4); plt.xlabel("Time (seconds)"); plt.ylabel("Amplitude"); plt.title("Waveform"); plt.tight_layout(); fig.savefig(waveform_path,dpi=150); plt.close(fig)
    spectrogram_path=output_dir/f"{stem}_spectrogram.png"; fig=plt.figure(figsize=(14,5)); plt.specgram(mono,NFFT=4096,Fs=sample_rate,noverlap=3072); plt.ylim(20,min(20000,sample_rate/2)); plt.xlabel("Time (seconds)"); plt.ylabel("Frequency (Hz)"); plt.title("Spectrogram"); plt.tight_layout(); fig.savefig(spectrogram_path,dpi=150); plt.close(fig); return [waveform_path,spectrogram_path]


def analyze_audio_v2_job(payload: dict) -> dict:
    artist=str(payload.get("artist") or "default"); song=str(payload.get("song") or "untitled"); job_dir=Path(payload.get("job_dir") or "/tmp/stemforge_analysis_v2"); job_dir.mkdir(parents=True,exist_ok=True); source=materialize_input(payload,"audio",job_dir/"source"); pcm=ensure_pcm_wav(source,job_dir/"source_pcm.wav"); audio,sample_rate=load_audio(pcm); analysis=analyze_array(audio,sample_rate); probe=ffprobe(source); report_path=job_dir/"analysis.json"; report_path.write_text(json.dumps({"artist":artist,"song":song,"probe":probe,"analysis":analysis},indent=2),encoding="utf-8"); outputs=[report_path]
    if bool(payload.get("render_plots",True)): outputs.extend(_build_plots(audio,sample_rate,job_dir,"analysis"))
    return {"status":"completed","engine":"StemForge full-song analysis v2","probe":probe,"analysis":analysis,"outputs":publish_files(outputs,artist=artist,category="analysis",ttl_seconds=int(payload.get("output_ttl_seconds",86400)))}


def compare_audio_v2_job(payload: dict) -> dict:
    artist=str(payload.get("artist") or "default"); job_dir=Path("/tmp/stemforge_compare_v2"); job_dir.mkdir(parents=True,exist_ok=True); source_path=materialize_input(payload,"source",job_dir/"source"); candidate_path=materialize_input(payload,"candidate",job_dir/"candidate"); source_pcm=ensure_pcm_wav(source_path,job_dir/"source.wav"); candidate_pcm=ensure_pcm_wav(candidate_path,job_dir/"candidate.wav"); source,sr=load_audio(source_pcm); candidate,csr=load_audio(candidate_pcm)
    if sr!=csr: raise RuntimeError("Internal sample-rate normalization failed.")
    length=min(len(source),len(candidate)); source=source[:length]; candidate=candidate[:length]; source_metrics=analyze_array(source,sr); candidate_metrics=analyze_array(candidate,sr); candidate_matched=candidate*(rms(source)/max(rms(candidate),1e-12)); difference=candidate_matched-source; similarity_db=10.0*np.log10((float(np.mean(np.square(source,dtype=np.float64)))+1e-20)/(float(np.mean(np.square(difference,dtype=np.float64)))+1e-20)); keys=("sample_peak_dbfs","true_peak_dbtp","rms_dbfs","crest_factor_db","integrated_lufs","stereo_correlation","transient_density_per_second"); differences={key:(round(float(candidate_metrics[key])-float(source_metrics[key]),4) if source_metrics.get(key) is not None and candidate_metrics.get(key) is not None else None) for key in keys}
    report={"status":"completed","listener_type":"objective DSP safety and similarity comparison","honesty_note":"This does not replace a human listening decision. It measures level, dynamics, spectrum, phase, timing, and source similarity.","source":source_metrics,"candidate":candidate_metrics,"differences":differences,"level_matched_residual_similarity_db":round(float(similarity_db),3),"safety":{"clipping_detected":candidate_metrics["clipped_sample_count"]>0,"true_peak_over_minus_0_8_dbtp":candidate_metrics["true_peak_dbtp"]>-0.8,"phase_risk":candidate_metrics["stereo_correlation"] is not None and candidate_metrics["stereo_correlation"]<-0.1,"crest_reduction_over_4db":differences["crest_factor_db"]<-4.0}}
    report_path=job_dir/"comparison.json"; report_path.write_text(json.dumps(report,indent=2),encoding="utf-8"); report["outputs"]=publish_files([report_path],artist=artist,category="comparisons",ttl_seconds=int(payload.get("output_ttl_seconds",86400))); return report
