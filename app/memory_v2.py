from __future__ import annotations

from datetime import datetime, timezone
from app.memory import get_memory, save_memory


def _now()->str:return datetime.now(timezone.utc).isoformat()

def record_rule_job(payload:dict)->dict:
    artist=str(payload.get("artist") or "sounddecay"); rule=str(payload.get("rule") or "").strip()
    if not rule:return {"status":"rejected","reason":"rule is required."}
    scope=str(payload.get("scope") or "global"); song=payload.get("song"); confidence=max(0.0,min(float(payload.get("confidence",1.0)),1.0)); source=str(payload.get("source") or "explicit_user_instruction"); memory=get_memory(artist); rules=memory.setdefault("structured_rules",[]); normalized=rule.lower().strip(); existing=next((item for item in rules if str(item.get("rule") or "").lower().strip()==normalized and item.get("scope")==scope and item.get("song")==song),None)
    if existing:
        existing["confirmations"]=int(existing.get("confirmations",1))+1; existing["confidence"]=max(float(existing.get("confidence",0.0)),confidence); existing["last_confirmed_at"]=_now()
    else:
        rules.append({"rule":rule,"scope":scope,"song":song,"confidence":confidence,"source":source,"confirmations":1,"created_at":_now(),"last_confirmed_at":_now()})
    path=save_memory(artist,memory); return {"status":"completed","artist":artist,"memory_path":str(path),"structured_rules":memory["structured_rules"]}

def get_production_profile_job(payload:dict)->dict:
    artist=str(payload.get("artist") or "sounddecay"); song=payload.get("song"); memory=get_memory(artist); rules=[item for item in memory.get("structured_rules",[]) if item.get("scope")=="global" or item.get("song")==song]
    return {"status":"completed","artist":artist,"song":song,"preferences":memory.get("preferences",{}),"applicable_rules":sorted(rules,key=lambda item:(item.get("scope")!="song",-float(item.get("confidence",0.0)),-int(item.get("confirmations",0)))),"feedback_count":len(memory.get("feedback",[])),"approved_lyric_timings":list(memory.get("approved_lyric_timings",{}).keys())}
