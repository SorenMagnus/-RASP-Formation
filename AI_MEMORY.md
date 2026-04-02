# AI_MEMORY - 褰撳墠鍛ㄦ湡浜ゆ帴鏂囨。

> 涓嬩竴浣?AI / 宸ョ▼甯堝惎鍔ㄥ悗锛屽厛瀹屾暣闃呰鏈枃浠讹紝鍐嶉槄璇?`PROMPT_SYSTEM.md` 涓?`RESEARCH_GOAL.md`锛岀劧鍚庡啀鍔ㄤ唬鐮併€? 
> 鏈枃浠舵寜 `2026-04-02` 鐨勭湡瀹炰粨搴撶姸鎬侀噸鍐欙紱鏃х増 AI_MEMORY 涓叧浜庢棫 `HEAD`銆佹棫 dirty/clean 鍙欒堪銆佷互鍙娾€滀笅涓€姝ュ彧鍘昏ˉ manifest-first audit鈥濈殑鎻忚堪閮藉凡缁忚繃鏈熴€?
---

## 0. 褰撳墠寮€鍙戞父鏍?
- 鏃ユ湡锛?  - `2026-04-02`
- Git 娓告爣锛?  - `HEAD = 54bb295fb6548c0167975510558ce2fff18251fc`
- 褰撳墠宸ヤ綔鏍戯細
  - 鍦ㄦ湰娆￠噸鍐欏墠锛宺epo-tracked 鏀瑰姩鍏?`3` 涓枃浠讹細
    - `AI_MEMORY.md`
    - `scripts/reproduce_paper.py`
    - `tests/test_reproduce_paper.py`
  - 鏈閲嶅啓瀹屾垚鍚庯紝`AI_MEMORY.md` 浠嶄細淇濇寔 modified
- 褰撳墠婧愮爜鏀瑰姩涓昏酱锛?  - 鏈疆鏂板鏀瑰姩宸蹭粠鈥渕anifest-first audit鈥濇帹杩涘埌鈥渃anonical runtime heartbeat / running-cell journal鈥?  - 涔熷氨鏄锛屽綋鍓?`reproduce_paper.py` 宸茬粡涓嶅彧鏄兘锛?    - 鐢熸垚 `manifest.json`
    - 鐢熸垚 `run_progress.json`
    - 鐢熸垚 `cell_progress.csv`
    - 鎵ц `--status-only / --validate-only`
  - 鐜板湪杩橀澶栬兘锛?    - 鐢熸垚 `run_runtime_state.json`
    - 鐢熸垚 `cell_runtime_state.csv`
    - 杩借釜 active cell 鐨?`runtime_status / started_at / last_heartbeat / heartbeat_age_seconds / stalled`

### 褰撳墠 live canonical snapshot

- 褰撳墠姝ｆ枃涓荤嚎浠嶇劧鏄細
  - `FSM + adaptive_apf + CBF-QP`
- `paper_canonical` 浠嶅彧鏈嶅姟浜?white-box 姝ｆ枃鐭╅樀
- 褰撳墠 `outputs/paper_canonical` 蹇呴』淇濈暀锛屼笉鑳藉垹鐩綍閲嶆潵
- 褰撳墠 canonical bundle 宸插垵濮嬪寲锛岃嚦灏戝凡瀛樺湪锛?  - `outputs/paper_canonical/manifest.json`
  - `outputs/paper_canonical/run_progress.json`
  - `outputs/paper_canonical/cell_progress.csv`
  - `outputs/paper_canonical/matrix_index.csv`
  - `outputs/paper_canonical/paper_acceptance.json`
  - `outputs/paper_canonical/run_runtime_state.json`
  - `outputs/paper_canonical/cell_runtime_state.csv`
- 鎴嚦鏈枃浠堕噸鍐欐椂鐨勭鐩樼湡鍊硷細
  - `bundle_progress = 0.0`
  - `bundle_completed_progress = 0.0`
  - `num_expected_cells = 55`
  - `num_complete_cells = 0`
  - `num_running_cells = 1`
  - `num_pending_cells = 54`
  - 褰撳墠浠嶇劧 **娌℃湁浠讳綍** `summary.csv` 钀界洏
- 褰撳墠 `run_runtime_state.json` 鐨?live snapshot 鏄剧ず锛?  - `running_cell_count = 1`
  - active cell 涓猴細
    - `scenario = s1_local_minima`
    - `variant_type = method`
    - `variant_name = no_rl`
    - `method = no_rl`
    - `run_id = s1_local_minima__no_rl`
  - active cell 褰撳墠涓猴細
    - `runtime_status = running`
    - `completed_seed_count = 0`
    - `completed_progress = 0.0`
    - `stalled = false`
- 褰撳墠杩涚▼鐪熷€硷細
  - 褰撳墠妫€娴嬪埌涓や釜 `python` 杩涚▼锛?    - `PID 5196`锛屽惎鍔ㄤ簬 `2026-04-02 13:03:12`锛孋PU 绱Н鏄捐憲锛屾洿鍍忓綋鍓?active `paper_canonical`
    - `PID 9412`锛屽惎鍔ㄦ洿鏃╋紝涓嶅簲榛樿瑙嗕负鏈疆 canonical 涓昏繘绋?  - 鍥犳锛屽綋鍓嶄笉鑳藉啀璇粹€滄病鏈夋椿鍔ㄤ腑鐨?reproduce_paper.py鈥?- 褰撳墠鏃ュ織鐘舵€侊細
  - `outputs/paper_canonical_run_stdout.log` 鐩墠浠嶄负绌?  - `outputs/paper_canonical_run_stderr.log` 褰撳墠浠嶅湪鍒凤細
    - `solver_status=maximum iterations reached`
    - `preview_violation_after_qp`
  - 鎴嚦杩欐閲嶅啓锛屾湭鐪嬪埌鑷村懡 traceback

### 褰撳墠鏈€閲嶈鐨勭粨璁?
- RL 杩欐潯绾跨殑宸ョ▼璇婃柇宸茬粡瓒冲锛?  - warm-start 涓?effective-threshold 鎸佷箙鍖栧凡缁忚瘉鏄?RL 鑳界湡瀹炰粙鍏?nominal layer
  - 浣?multi-seed 鏁堢巼浠嶄笉浼樹簬 `no_rl`
  - 鍥犳 RL 缁х画鐣欏湪闄勫綍澧炲己鍊欓€夛紝涓嶅啀缁戞灦姝ｆ枃涓荤嚎
- 褰撳墠鐪熸杩樻病鏀跺彛鐨勶紝涓嶆槸绠楁硶锛屼篃涓嶆槸 canonical sealing锛岃€屾槸锛?  - **white-box canonical 闀胯窇鐨勬渶缁?artifact 杩樻病鏈夌湡姝ｈ惤鐩?*
- 褰撳墠鏈€鍏抽敭鐨勬柊澧炶兘鍔涘凡缁忎笉鏄€滆兘涓嶈兘楠屾敹鈥濓紝鑰屾槸鈥滃湪绗竴涓?`summary.csv` 钀界洏涔嬪墠锛岃兘鍚︾湅娓?run 鏄湪娲荤潃銆佸崱浣忋€佽繕鏄凡缁忓け鑱斺€?- 杩欒疆 heartbeat / runtime journal 浠ｇ爜宸茬粡鎶娾€滃彲瑙傛祴鎬х己鍙ｂ€濊ˉ涓婁簡
- 浣嗗綋鍓?bundle 浠嶇劧鍙槸锛?  - 宸插垵濮嬪寲
  - 姝ｅ湪璺戠涓€涓?cell
  - 灏氭棤浠讳綍瀹屾暣 cell 瀹屾垚

---

## 1. 宸插畬鎴愬伐浣?
### 1.1 鍘嗗彶涓荤嚎鑳藉姏浠嶇劧鎴愮珛

- gatefix 宸插畬鎴愶細
  - `confidence_raw` 浣跨敤 Beta 鏂瑰樊鏍″噯
  - 涓ら槇鍊兼粸鍥?gate 宸插湪浣?- reward_v2 宸插畬鎴愶細
  - reward shaping
  - reward 閰嶇疆鍖?  - PPO reward diagnostics
- training warm-start 宸插畬鎴愶細
  - `tau_enter_start = 0.25`
  - `tau_exit_start = 0.15`
  - `gate_warmup_timesteps = 20000`
- effective runtime threshold 宸插畬鎴愬苟宸茶疮閫氬埌锛?  - runtime diagnostics
  - replay
  - RL attribution
- RL 璺嚎宸茬粡瀹屾垚鈥滃畠鑳藉惁鐪熷疄浠嬪叆 nominal layer鈥濈殑宸ョ▼璇婃柇浠诲姟锛?  - 鑳借繘 gate
  - 浣?multi-seed 缁撴灉浠嶄笉浼樹簬 white-box baseline

### 1.2 manifest-first canonical audit 宸插畬鎴?
- `scripts/reproduce_paper.py`
  - 宸叉敮鎸侊細
    - `--status-only`
    - `--validate-only`
  - 浜岃€呭綋鍓嶉兘宸茬粡鏄?manifest-first锛?    1. 鑻?`outputs/<exp_id>/manifest.json` 瀛樺湪锛屽垯浼樺厛浠?manifest 鎭㈠ canonical spec
    2. manifest 涓嶅瓨鍦ㄤ笖鏄惧紡浼犱簡 `--canonical-matrix` 鏃讹紝鎵嶅厑璁?fallback 鍒?canonical 甯搁噺
    3. manifest 涓嶅瓨鍦ㄤ笖鏈樉寮忎紶 `--canonical-matrix` 鏃讹紝杩斿洖闈為浂閫€鍑虹爜 `2`
- 绾鐩樺璁″綋鍓嶅凡鍏峰锛?  - `manifest.json`
  - `run_progress.json`
  - `cell_progress.csv`
  - `matrix_index.csv`
  - `paper_acceptance.json`
- 杩欎簺浜х墿鐜板湪閮藉彲鍦?partial bundle 涓婂埛鏂帮紝涓嶅啀渚濊禆 CLI 榛樿 `seeds/scenarios/methods/ablations`

### 1.3 鏈疆鏂板婧愮爜瀹炵幇锛歝anonical runtime heartbeat / running-cell journal

- `scripts/reproduce_paper.py`
  - 宸叉柊澧炶繍琛屾€佸父閲忎笌杩愯鎬佹枃浠跺啓鍏ラ€昏緫
  - 宸叉柊澧炲苟钀藉湴涓ょ被杩愯鎬佷骇鐗╋細
    - `run_runtime_state.json`
    - `cell_runtime_state.csv`
  - 宸插疄鐜颁互涓嬭繍琛屾€佸瓧娈碉細
    - `runtime_status 鈭?{pending, running, complete, failed}`
    - `started_at`
    - `last_heartbeat`
    - `finished_at`
    - `heartbeat_age_seconds`
    - `stalled`
    - `completed_seed_count`
    - `completed_progress`
  - 宸插疄鐜?heartbeat 绾跨▼锛?    - 蹇冭烦鍛ㄦ湡 `30` 绉?    - 浠呭埛鏂?runtime journal
    - 涓嶄慨鏀?acceptance 鍙ｅ緞
  - 宸插疄鐜颁覆琛岃繍琛岀害鏉燂細
    - 浠绘剰鏃跺埢鍙厑璁?`running_cell_count 鈭?{0,1}`
    - 鑻ョ鐩樹笂鍑虹幇澶氫釜 `running` rows锛屽垯鍙繚鐣欐渶鏂?heartbeat 鐨勯偅涓负 `running`
  - 宸插疄鐜?`status-only` 鐨勮繍琛屾€佹憳瑕佽緭鍑猴細
    - `bundle_completed_progress`
    - `running_cell_count`
    - `num_pending_cells`
    - `num_running_cells`
    - `num_complete_cells`
    - `num_failed_cells`
    - active cell 鐨?`scenario / variant_type / variant_name / method / run_id / heartbeat_age_seconds / stalled`
  - 宸蹭繚鎸?`validate-only` 鐨?acceptance 涓嶅彉鎬э細
    - 浠嶇劧鍙敱 `summary.csv` 鍜?bundle sealing 鍐冲畾 `bundle_complete`
    - runtime `running` 缁濅笉浼氳褰撴垚 `complete`

- `tests/test_reproduce_paper.py`
  - 鏈疆鏂板骞惰鐩栦簡 heartbeat / runtime journal 鐩稿叧閾捐矾锛?    - 杩愯鎬佹枃浠剁敓鎴愭祴璇?    - 鍗?running cell 绾︽潫娴嬭瘯
    - `stalled` 鍒ゅ畾娴嬭瘯
    - `status-only` 绾鐩樿繍琛屾€佽緭鍑烘祴璇?    - acceptance 涓嶅彉鎬ф祴璇?    - `status-only` 骞傜瓑鎬ф祴璇曪紝鐜板凡鎶婏細
      - `run_runtime_state.json`
      - `cell_runtime_state.csv`
      绾冲叆蹇収姣斿

### 1.4 鏈疆宸插畬鎴愮殑楠岃瘉

- 宸?fresh 閫氳繃锛?  - `python -m compileall src tests scripts`
  - `python -m pytest -q tests/test_stats_export.py tests/test_reproduce_paper.py`
- 褰撳墠鏈疆鏈€鍙俊鐨?fresh 楠岃瘉缁撴灉锛?  - `20 passed`
- 鏈疆娌℃湁 fresh rerun 瀹屾暣锛?  - `python -m pytest -q`
- 鍥犳褰撳墠涓嶈兘瀹ｇО锛?  - 鈥滄湰杞?full-suite fresh rerun 鍏ㄧ豢鈥?
### 1.5 鏈疆鐪熷疄杩愯楠岃瘉宸插畬鎴?
- 杩欒疆 heartbeat 浠ｇ爜涓嶅彧鍋滅暀鍦ㄥ崟鍏冩祴璇曪紝宸茬粡瀵圭湡瀹?partial bundle 鍋氳繃鍛戒护绾块獙璇侊細
  - `python scripts/reproduce_paper.py --exp-id paper_canonical --status-only`
  - `python scripts/reproduce_paper.py --exp-id paper_canonical --validate-only`
- 鐪熷疄缁撴灉鏄細
  - `status-only` 鐜板湪鑳藉湪 partial bundle 涓婃墦鍗?live runtime 鎽樿
  - `validate-only` 鍦ㄥ綋鍓?bundle 鏈畬鎴愭椂锛屼粛鎸夐鏈熻繑鍥炩€滀笉閫氳繃鈥?  - 杩欒瘉鏄庯細
    - heartbeat 浠ｇ爜宸叉帴鍏ョ湡瀹?canonical 闀胯窇
    - acceptance 瑙勫垯娌℃湁琚?runtime journal 姹℃煋

---

## 2. 褰撳墠鐮旂┒鍒ゆ柇

- 褰撳墠椤圭洰鍓╀笅鐨勪富闂锛屽凡缁忎笉鏄€滆繕缂哄摢娈垫牳蹇冪畻娉曚唬鐮佲€?- 褰撳墠鐪熸鏈敹鍙ｇ殑鏄細
  - **white-box canonical artifact 杩樻病鏈夎窇瀹?*
- 鏇村叿浣撳湴璇达細
  - canonical audit 宸插畬鎴?  - runtime heartbeat 涔熷凡瀹屾垚
  - 浣嗗綋鍓?bundle 浠嶅仠鍦ㄧ涓€涓?cell
  - 灏氭棤浠讳綍 `summary.csv`
  - 鍥犳 `paper_acceptance.json` 浠嶇劧鍙兘缁欏嚭鈥滄湭瀹屾垚鈥?
褰撳墠鍒ゆ柇瑕佺偣濡備笅锛?
- RL锛?  - 褰撳墠涓嶅啀鏄富浠诲姟
  - 杩戞湡涓嶈缁х画娑堣€楀伐绋嬫椂闂村湪 RL 涓?- canonical 浠ｇ爜锛?  - 鈥滆兘鍚︾湅瑙?run 鍦ㄨ窇浠€涔堚€濊繖浠朵簨锛屽凡缁忛€氳繃 runtime journal 瑙ｅ喅
  - 褰撳墠娌℃湁蹇呰鍐嶉噸鍐?manifest-first audit
  - 褰撳墠涔熶笉闇€瑕佸啀鍥炲幓纰?reward / gate / warm-start
- 鍓╀綑涓昏椋庨櫓宸茶浆鍖栦负锛?  - 闀挎椂闂?canonical 杩愯鐨勫悶鍚?/ stall / 涓€斿け鑱旇瘖鏂?  - 浠ュ強 canonical 鏈€缁堟槸鍚﹁兘涓€娆℃€ф弧瓒筹細
    - `bundle_complete = true`
    - `primary_safety_valid = true`

鎹㈠彞璇濊锛?
- 鐜板湪宸茬粡闈炲父鎺ヨ繎鈥滀唬鐮佹敹鍙ｂ€?- 浣嗚窛绂烩€滃彲浠ユ斁蹇冭姝ｆ枃 artifact 宸插畬鎴愨€濊繕宸細
  - canonical 鐪熸璺戝畬
  - `paper_acceptance.json` 杩囨渶缁堥獙鏀?
---

## 3. 涓嬩竴姝ユ寚浠?
### 3.1 鎬诲師鍒?
涓嬩竴浣嶅伐绋嬪笀鍚姩 AI 鍚庯細

- 涓嶈鍐嶆敼 RL reward
- 涓嶈鍐嶆敼 RL gate
- 涓嶈鍐嶆敼 warm-start 鍏紡
- 涓嶈鍐嶉噸鍐?manifest-first audit
- 涓嶈鍒?`outputs/paper_canonical`
- 涓嶈鎶婂綋鍓?partial bundle 褰撴垚鍨冨溇鐩綍娓呯悊鎺?- 涓嶈鎶?runtime journal 褰撴垚 acceptance 鏇夸唬鍝?
涓嬩竴鏉＄珛鍗虫墽琛岀殑浠ｇ爜浠诲姟锛屽簲鍥寸粫锛?
- **canonical runtime process-aware liveness probe / orphan reconciliation**

灞曞紑锛屼篃灏辨槸鍦?heartbeat 涔嬩笂锛屽啀琛モ€滆繘绋嬬骇娲绘€х‘璁も€濓紝璁╃郴缁熻兘鍖哄垎锛?
- heartbeat 姝ｅ父銆佽繘绋嬩篃娲荤潃
- heartbeat 杩囦箙鏈洿鏂帮紝浣嗚繘绋嬭繕娲荤潃
- heartbeat 杩囦箙鏈洿鏂帮紝涓斿啓 heartbeat 鐨勮繘绋嬪凡缁忎笉瀛樺湪

### 3.2 绔嬪嵆鎵ц鐨勪唬鐮佷换鍔?
鍦ㄤ互涓嬫枃浠朵腑缁х画瀹炵幇锛?
- `scripts/reproduce_paper.py`
- `tests/test_reproduce_paper.py`

濡傜‘鏈夊繀瑕侊紝鍐嶆渶灏忓閲忚Е纰帮細

- `src/apflf/analysis/stats.py`
- `tests/test_stats_export.py`

浼樺厛澶嶇敤褰撳墠宸叉湁鑳藉姏锛?
- 褰撳墠 heartbeat thread
- 褰撳墠 `run_runtime_state.json`
- 褰撳墠 `cell_runtime_state.csv`
- 褰撳墠 `status-only / validate-only`
- 褰撳墠 `manifest.json`
- 褰撳墠 `run_progress.json`
- 褰撳墠 `cell_progress.csv`

涓嶈鍙﹁捣涓€濂楀钩琛?watchdog 鑴氭湰銆?
### 3.3 杩欐浠ｇ爜蹇呴』鏂板浠€涔?
鍦ㄨ繍琛屾€佷骇鐗╀腑缁х画鏂板骞剁淮鎶や互涓嬪瓧娈碉細

- `runner_pid`
- `runner_started_at`
- `process_alive`
- `orphaned`

寤鸿鑷冲皯钀藉湪锛?
- `run_runtime_state.json`
- `cell_runtime_state.csv`

鍏朵腑锛?
- `runner_pid`锛?  - 褰撳墠璐熻矗璺戣 cell 鐨?Python 杩涚▼ PID
- `runner_started_at`锛?  - 璇ヨ繘绋嬬殑鍚姩鏃堕棿鎴?- `process_alive`锛?  - 褰撳墠 audit 鏃讹紝纾佺洏璁板綍鐨?`runner_pid` 鏄惁浠嶅搴斿悓涓€涓椿杩涚▼
- `orphaned`锛?  - 璇?cell 浠嶆樉绀?`runtime_status=running`锛屼絾璁板綍涓殑杩涚▼宸蹭笉鍐嶅瓨娲?
### 3.4 蹇呴』婊¤冻鐨勭姸鎬佹満涓庢暟瀛︾害鏉?
璁帮細

- `S = manifest.expected_seeds`
- `C = manifest.expected_cells`
- `|S| = 30`
- `|C| = 55`
- `H = 900` 绉?- 杩涚▼鍚姩鏃堕棿鍖归厤瀹瑰樊锛?  - `螖 = 5` 绉?
瀵逛换鎰忔椂鍒?`t`銆佷换鎰?cell `c 鈭?C`锛屽畾涔夛細

- `completed_seed_count_t(c)`锛?  - 浠庣鐩?`summary.csv` 涓鍑虹殑宸插畬鎴?seed 鏁?- `completed_progress_t(c) = completed_seed_count_t(c) / |S|`
- `bundle_completed_progress_t = (1 / (|C| * |S|)) * 危_c completed_seed_count_t(c)`
- `heartbeat_age_t(c) = max(0, now_t - last_heartbeat_t(c))`
- `process_alive_t(c) = 1[瀛樺湪杩涚▼ p锛屼娇寰?pid(p)=runner_pid(c) 涓?|start_time_utc(p)-runner_started_at(c)| <= 螖]`
- `stalled_t(c) = 1[runtime_status_t(c)=running and heartbeat_age_t(c) > H]`
- `orphaned_t(c) = 1[runtime_status_t(c)=running and process_alive_t(c)=0]`

蹇呴』婊¤冻锛?
- `0 <= completed_seed_count_t(c) <= |S|`
- `0 <= completed_progress_t(c) <= 1`
- `0 <= bundle_completed_progress_t <= 1`
- 鍦ㄤ笉鍒犻櫎宸叉湁缁撴灉鐨勫墠鎻愪笅锛?  - `completed_seed_count_t(c)` 瀵?`t` 鍗曡皟涓嶅噺
  - `completed_progress_t(c)` 瀵?`t` 鍗曡皟涓嶅噺
  - `bundle_completed_progress_t` 瀵?`t` 鍗曡皟涓嶅噺
- `running_cell_count_t 鈭?{0, 1}`
- `process_alive_t(c) 鈭?{0,1}`
- `stalled_t(c) 鈭?{0,1}`
- `orphaned_t(c) 鈭?{0,1}`

鐘舵€佽涔夊浐瀹氫负锛?
- 鍒氳繘鍏ヨ繍琛岋細
  - `runtime_status = running`
  - `started_at = last_heartbeat`
  - `runner_pid` 鍜?`runner_started_at` 蹇呴』鍐欏叆
- 姝ｅ父瀹屾垚锛?  - `runtime_status: running -> complete`
  - `finished_at` 鍐欏叆
  - `process_alive` 鍙负 `0` 鎴栫┖锛屼笉鍙備笌 complete 鍒ゅ畾
- 寮傚父閫€鍑猴細
  - `runtime_status: running -> failed`
  - `finished_at` 鍐欏叆
- 鑻ュ閮?kill 鎴栨墜鍔ㄥ仠姝㈠鑷存棤娉曟樉寮忓啓鍏?`failed`锛?  - 涓嶅己琛屼吉閫?`failed`
  - 鐢?`status-only` 閫氳繃锛?    - `stalled=true`
    - `process_alive=false`
    - `orphaned=true`
    鏉ユ毚闇测€滃け鑱斾腑鐨?running cell鈥?
閲嶈 acceptance 绾︽潫缁х画鍥哄畾涓猴細

- `validate-only` 浠嶇劧鍙兘鎸?`summary.csv` 涓?sealing 缁撴灉鍒ゅ畾
- 杩愯涓€乻talled銆乷rphaned 鐨?cell 閮?**涓嶈兘** 琚畻鎴?`complete`
- `primary_safety_valid` 瑙勫垯淇濇寔瀹屽叏涓嶅彉

### 3.5 涓嬩竴杞祴璇曡姹?
蹇呴』鏂板骞堕€氳繃浠ヤ笅娴嬭瘯锛?
- process-aware runtime 瀛楁鐢熸垚娴嬭瘯锛?  - `run_runtime_state.json`
  - `cell_runtime_state.csv`
  涓繀椤诲嚭鐜帮細
  - `runner_pid`
  - `runner_started_at`
  - `process_alive`
  - `orphaned`
- liveness 鍒ゅ畾娴嬭瘯锛?  - 褰?`runner_pid` 瀵瑰簲杩涚▼瀛樺湪涓斿惎鍔ㄦ椂闂村尮閰嶆椂锛?    - `process_alive = true`
  - 鍚﹀垯锛?    - `process_alive = false`
- orphan 鍒ゅ畾娴嬭瘯锛?  - 褰?`runtime_status = running` 涓?`process_alive = false` 鏃讹細
    - `orphaned = true`
- stall 涓?orphan 瑙ｈ€︽祴璇曪細
  - `heartbeat_age_seconds > 900` 鏃讹紝`stalled = true`
  - `stalled` 涓?`orphaned` 涓嶈兘浜掔浉鏇夸唬
- `status-only` 绾鐩樺箓绛夋€ф祴璇曪細
  - 瀵瑰悓涓€ bundle 杩炵画鎵ц涓ゆ `--status-only`
  - 浠ヤ笅浜х墿鏁版嵁鍐呭蹇呴』涓€鑷达細
    - `run_progress.json`
    - `cell_progress.csv`
    - `run_runtime_state.json`
    - `cell_runtime_state.csv`
    - `matrix_index.csv`
    - `paper_acceptance.json`
- acceptance 涓嶅彉鎬ф祴璇曪細
  - 鍗充娇 `runtime_status=running`
  - 鍗充娇 `process_alive=true`
  - 鍗充娇 `orphaned=false`
  - 鑻?`summary.csv` 灏氭湭瀹屾暣锛屽垯 `validate-only` 浠嶅繀椤诲け璐?
### 3.6 涓嬩竴杞繍琛屼笌楠屾敹椤哄簭

涓嬩竴浣嶅伐绋嬪笀搴旀寜杩欎釜椤哄簭缁х画锛?
1. 鍏堟鏌ュ綋鍓?`paper_canonical` 鏄惁浠嶅湪杩愯
2. 鑻ヤ粛鍦ㄨ繍琛岋紝涓嶈鍒犻櫎褰撳墠 `outputs/paper_canonical`
3. 鍏堝疄鐜?process-aware liveness / orphan reconciliation
4. 閫氳繃浠ヤ笅楠岃瘉锛?   - `python -m compileall src tests scripts`
   - `python -m pytest -q tests/test_stats_export.py tests/test_reproduce_paper.py`
5. 鐒跺悗缁х画/鎭㈠姝ｆ枃 canonical锛?   - `python scripts/reproduce_paper.py --exp-id paper_canonical --canonical-matrix --skip-existing`
6. 鍐嶆墽琛岋細
   - `python scripts/reproduce_paper.py --exp-id paper_canonical --status-only`
   - `python scripts/reproduce_paper.py --exp-id paper_canonical --validate-only`

鏈€缁堥獙鏀舵爣鍑嗕粛鍥哄畾涓猴細

- `outputs/paper_canonical` 瀛樺湪
- `run_progress.json` 鏈€缁?`bundle_progress = 1.0`
- `paper_acceptance.json` 涓?`bundle_complete = true`
- `paper_acceptance.json` 涓?`primary_safety_valid = true`
- 鎵€鏈?expected canonical cells 瑕嗙洊 `30` seeds

---

## 4. 鎶€鏈爤绾㈢嚎

浠ヤ笅瑕佹眰蹇呴』缁х画涓ユ牸淇濈暀锛屼笉寰楄鍚庣画宸ョ▼甯堢牬鍧忥細

### 4.1 涓绘灦鏋勭孩绾?
- 姝ｆ枃涓荤嚎濮嬬粓鏄櫧鐩掍富閾撅細
  - `FSM + adaptive_apf + CBF-QP`
- `paper_canonical` 鍙湇鍔′簬 white-box 姝ｆ枃鐭╅樀
- RL 涓嶅緱閲嶆柊鍗囩骇涓哄綋鍓嶄富绾?
### 4.2 RL 绾㈢嚎

- RL 浠嶄弗鏍奸檺鍒朵负锛?  - `param-only supervisor`
- 涓嶅厑璁稿洖婊氭垚锛?  - `mode-only RL`
  - `full supervisor continuous control`
- 涓嶅厑璁歌 RL 鐩存帴杈撳嚭杩炵画鎺у埗閲?- 涓嶅厑璁稿紩鍏?`SB3`
- `PROMPT_SYSTEM.md` 涓?`RESEARCH_GOAL.md` 閲屼繚鐣欑殑鏃?`mode-only RL` 鐞嗚妯℃澘涓嶆槸褰撳墠娲诲绾?
### 4.3 鍏叡鎺ュ彛绾㈢嚎

浠ヤ笅鎺ュ彛涓嶅緱淇敼锛?
- `ModeDecision(mode, theta, source, confidence)`
- `compute_actions(observation, mode, theta=None)`

### 4.4 瀹夊叏灞傜孩绾?
- 涓嶅厑璁镐慨鏀?safety red-line 鏂囦欢
- 涓嶅厑璁镐负浜嗚拷姹傛晥鐜囪€屾斁鏉?CBF-QP 鐨?safety acceptance 鍙ｅ緞

### 4.5 缁熻涓庤鏂囧彛寰勭孩绾?
- 榛樿涓昏〃缁熻缁х画浣跨敤 deterministic bootstrap CI
- `paper_acceptance.json` 蹇呴』缁х画浣滀负姝ｆ枃 artifact 鐨勫敮涓€鏀跺彛鍏ュ彛
- paired delta 缁х画鍙厑璁告寜鐩稿悓 seed 閰嶅

---

## 5. 涓€鍙ヨ瘽浜ゆ帴

褰撳墠鏈€閲嶈鐨勪簨瀹炴槸锛?
- manifest-first canonical audit 宸插畬鎴?- runtime heartbeat / running-cell journal 宸插畬鎴?- `paper_canonical` 褰撳墠姝ｅ湪璺戠涓€涓?cell锛屼絾浠嶆棤浠讳綍 `summary.csv`
- RL 宸茬粡閫€鍑轰富浠诲姟闃熷垪
- 涓嬩竴浣嶅伐绋嬪笀涓嶈鍐嶇绠楁硶锛岃€屽簲绔嬪埢琛?**process-aware liveness / orphan reconciliation**锛岀劧鍚庣户缁妸 `paper_canonical` 璺戝埌 `bundle_complete = true`
