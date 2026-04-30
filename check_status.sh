#!/usr/bin/env bash
# RT-DETR P2-feedback HPC training status dashboard (v2).
# Tracks the v2 retrain — gate_init=0, gate_floor=0.1, P2/P3-only feedback.
# Pulls live data from Bocconi HPC and prints a clean summary.
#
# Usage:
#   bash check_status.sh
#
# Requires:
#   - Bocconi GlobalProtect VPN up (so 10.35.5.3 is routable)
#   - SSH key auth or interactive password to 3415496@10.35.5.3
#   - python3 in PATH

set -u
HPC="3415496@10.35.5.3"
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"

# --- pull raw data over one ssh session ---
RAW=$($SSH "$HPC" 'echo "===QUEUE==="
squeue -u $USER -o "%.10i|%.2t|%.10M|%R" -h 2>/dev/null
echo "===LOG==="
cat /home/3415496/rt-detr/output/rtdetr_r50vd_hpc_feedback_p2_v2/log.txt 2>/dev/null
echo "===PROG==="
LATEST=$(ls -t ~/rt-detr/p2_v2_chain_*.out 2>/dev/null | head -1)
echo "FILE=$LATEST"
tail -3 "$LATEST" 2>/dev/null
echo "===INIT_LR==="
grep "Initial lr:" "$LATEST" 2>/dev/null | tail -1
echo "===GATE==="
for f in ~/rt-detr/p2_v2_chain_*.out; do grep "feedback:" "$f" 2>/dev/null; done | tail -5
echo "===QUOTA==="
lquota 2>&1 | grep "USED=" | head -1
echo "===CKPT==="
ls -la /home/3415496/rt-detr/output/rtdetr_r50vd_hpc_feedback_p2_v2/*.pth 2>/dev/null | awk "{print \$5/1e6\" MB \"\$9}"' 2>/dev/null) || { echo "ERROR: cannot reach HPC. VPN up?"; exit 1; }

python3 - "$RAW" <<'PY'
import json, re, sys, datetime
raw = sys.argv[1]
sections = {}
cur = None
for line in raw.splitlines():
    m = re.match(r'^===(\w+)===', line)
    if m:
        cur = m.group(1); sections[cur] = []
    elif cur is not None:
        sections[cur].append(line)

q       = '\n'.join(sections.get('QUEUE',   [])).strip()
prog    = '\n'.join(sections.get('PROG',    [])).strip()
log     = '\n'.join(sections.get('LOG',     [])).strip()
gate    = '\n'.join(sections.get('GATE',    [])).strip()
quota   = '\n'.join(sections.get('QUOTA',   [])).strip()
ckpt    = '\n'.join(sections.get('CKPT',    [])).strip()
init_lr = '\n'.join(sections.get('INIT_LR', [])).strip()

# queue
state, jid, nodelist, elapsed = 'NO JOB', '-', '-', '-'
if q:
    parts = q.split('|')
    if len(parts) >= 4:
        jid, st, elapsed, nodelist = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
        state = {'R':'RUNNING','PD':'PENDING','CG':'COMPLETING','CD':'COMPLETED'}.get(st, st)

# current iter
m = re.search(r'Epoch: \[(\d+)\] +\[ *(\d+)/(\d+)\] +eta: ([^ ]+) +lr: ([^ ]+) +loss: ([\d.]+) \(([\d.]+)\)', prog)
cur_epoch = cur_iter = total_iter = eta = lr = loss_inst = loss_avg = None
if m:
    cur_epoch, cur_iter, total_iter = int(m.group(1)), int(m.group(2)), int(m.group(3))
    eta, lr, loss_inst, loss_avg = m.group(4), m.group(5), m.group(6), m.group(7)

# AP per epoch
aps = []
for line in log.splitlines():
    line = line.strip()
    if not line: continue
    try:
        d  = json.loads(line)
        ap = d['test_coco_eval_bbox']
        aps.append({'ap':ap[0]*100, 'aps':ap[3]*100, 'apm':ap[4]*100, 'apl':ap[5]*100})
    except Exception:
        pass

# latest gate
gate_val = None
glines = [l for l in gate.splitlines() if l.strip()]
if glines:
    gm = re.search(r'gate=([\d.]+)', glines[-1])
    if gm:
        gate_val = gm.group(1)

W = 60
def hr(c='-'): print(c * W)
def kv(k, v): print(f'  {k:<14} {v}')

now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
hr('=')
print(f' RT-DETR P2-feedback v2 HPC dashboard  {now}')
print(f'   (gate_init=0.0, gate_floor=0.1, P2/P3-only feedback)')
hr('=')

print('\nJOB')
kv('ID',      jid)
kv('State',   state)
kv('Elapsed', elapsed)
kv('Node',    nodelist)
kv('Quota',   quota.replace('USED=', '') if quota else '-')

if cur_epoch is not None:
    pct = 100.0 * cur_iter / total_iter
    print('\nLIVE TRAINING')
    kv('Internal ep', cur_epoch)
    kv('Iter',        f'{cur_iter}/{total_iter} ({pct:.1f}%)')
    kv('ETA',         eta)
    # show LR in scientific to avoid trailing-zero trunc
    try:
        kv('LR (group 0)', f'{float(lr):.2e}  (raw: {lr})')
    except Exception:
        kv('LR (group 0)', lr)
    kv('Loss inst',   loss_inst)
    kv('Loss avg',    loss_avg)
    if gate_val:
        kv('Gate',    gate_val)

# Initial LRs across all param groups (printed at job start; survives full precision)
if init_lr:
    m = re.search(r'\[([^\]]+)\]', init_lr)
    if m:
        try:
            lrs = [float(x.strip()) for x in m.group(1).split(',')]
            print('\nLR PER PARAM GROUP (job start, full precision)')
            for i, v in enumerate(lrs):
                names = ['backbone (group 0)', 'feedback (group 1)', 'enc-norm (group 2)',
                         'dec-norm (group 3)', 'catchall (group 4)']
                label = names[i] if i < len(names) else f'group {i}'
                print(f'  {label:<25} {v:.2e}')
            # diagnose whether decay is applied vs the un-decayed values
            base_known = 5e-5  # un-decayed base lr (chain 1 / chain 2)
            decayed_known = 5e-6
            if len(lrs) >= 5:
                if abs(lrs[4] - decayed_known) / decayed_known < 0.05:
                    print('  -> 10× LR decay IS applied vs chain 1/2 baseline')
                elif abs(lrs[4] - base_known) / base_known < 0.05:
                    print('  -> running at the original chain-1/2 LR (no decay)')
        except Exception:
            print('\nLR PER PARAM GROUP (raw)')
            print('  ' + init_lr)

print('\nAP_S TRAJECTORY (val2017 @ 640)')
COCO_BASELINE_APS = 34.7   # vanilla RT-DETR-R50 paper number
V1_FINAL_APS      = 33.9   # v1 feedback ON @ 640 (12 ep)
prev = None
for i, e in enumerate(aps):
    delta = (e['aps'] - prev) if prev is not None else None
    delta_str = f' {delta:+.1f}' if delta is not None else '     '
    print(f"  ep {i:>2}:  AP={e['ap']:5.1f}  AP_S={e['aps']:5.1f}{delta_str}  AP_M={e['apm']:5.1f}  AP_L={e['apl']:5.1f}")
    prev = e['aps']

if aps:
    last = aps[-1]['aps']
    print(f'\n  vs RT-DETR baseline (34.7):     {last - COCO_BASELINE_APS:+.1f}')
    print(f'  vs v1 feedback final (33.9):    {last - V1_FINAL_APS:+.1f}')
    print(f'  Distance to +1.0 target (35.7): {last - 35.7:+.1f}')

print('\nSUCCESS METRIC (final, post-eval): AP_S(feedback ON) > AP_S(feedback OFF)')
print('  -> the v2 retrain is meant to make this gap > 0.')

print('\nCKPTS')
for line in ckpt.splitlines():
    print('  ' + line)

hr('=')
PY
