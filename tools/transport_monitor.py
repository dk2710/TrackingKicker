
import json, time, urllib.request
from pathlib import Path
BASE='http://127.0.0.1:5011'
OUT=Path('/home/dkleis/dev/tischkicker/transport_only_ramp90_180_results.jsonl')
OUT.write_text('', encoding='utf-8')

def request(path, method='GET', payload=None):
    data=None; headers={}
    if payload is not None:
        data=json.dumps(payload).encode(); headers['Content-Type']='application/json'
    try:
        with urllib.request.urlopen(urllib.request.Request(BASE+path,data=data,headers=headers,method=method),timeout=20) as response:
            return json.loads(response.read().decode())
    except Exception as exc:
        return {'request_error':repr(exc)}

def record(phase, elapsed, extra=None):
    data=request('/api/status')
    keep={k:data.get(k) for k in (
        'camera_open','grabbing','requested_fps','resulting_fps','camera_fps',
        'camera_frames_total','failed_grabs','payload_errors','dropped_frames',
        'reconnects','frame_ipc','frame_publishing_enabled','last_error',
        'last_failure_phase','last_result_diagnostic','last_success_block_id',
        'last_frame_age_seconds')}
    keep.update({'phase':phase,'elapsed_seconds':round(elapsed,1),'time':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    if extra: keep.update(extra)
    with OUT.open('a',encoding='utf-8') as h: h.write(json.dumps(keep,ensure_ascii=False)+'\n')
    return keep

def set_fps(fps):
    response=request('/api/settings','POST',{'camera':{'force_fps':fps}})
    with OUT.open('a',encoding='utf-8') as h:
        h.write(json.dumps({'phase':'switch','target_fps':fps,'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'response':response},ensure_ascii=False)+'\n')
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        status=request('/api/status')
        if status.get('camera_open') and abs(float(status.get('requested_fps',0))-fps)<0.1 and abs(float(status.get('resulting_fps',0) or 0)-fps)<1.0:
            return
        time.sleep(2)

for fps in (90,120,150,180):
    set_fps(fps)
    phase_start=time.monotonic()
    record(f'{fps}_fps',0)
    while time.monotonic()-phase_start < 120:
        time.sleep(10)
        record(f'{fps}_fps',time.monotonic()-phase_start)
    record(f'{fps}_fps',120)
with OUT.open('a',encoding='utf-8') as h: h.write(json.dumps({'phase':'complete','time':time.strftime('%Y-%m-%dT%H:%M:%S%z')})+'\n')
