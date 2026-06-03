富场景全双工用例（预合成 WAV）
================================

音色（默认 Edge TTS，需 pip install edge-tts + ffmpeg）:
  - 问候、助手回复: 女声 zh-CN-XiaoxiaoNeural
  - 用户句: 男声 zh-CN-YunxiNeural
  - 语速 -5%、音高 +1Hz，略更口语化（可在 scripts/prepare_duplex_scenario_wavs.py 顶部常量修改）

1) 生成/覆盖 wavs/::

     .venv\Scripts\python.exe scripts\prepare_duplex_scenario_wavs.py --force

   仅 pyttsx3（无男女分音色）::

     .venv\Scripts\python.exe scripts\prepare_duplex_scenario_wavs.py --engine pyttsx3 --force

2) 脚本化回放::

     .venv\Scripts\python.exe examples\full_duplex_scripted_rich.py --device xpu --call-audio

3) _还原.txt 中「用户（稿）」= scenario.json 文案，与 WAV 一致；〔ASR〕为识别结果供对照。

4) 编辑 scenario.json 后请重新执行步骤 1。

5) 节奏（觉得「衔接偏慢」可调 scenario.json，单位见键名）::
     - silence_after_greeting_ms / silence_between_turns_ms：问候后、各轮助手播完后的静音
     - greeting_tail_extra_chunks：问候 TTS 播完后多推的静音块数（chunk_ms 与 chunk_ms 字段一致）
     - post_user_silence_ms + post_user_vad_extra_chunks：用户句推完后等 VAD/分段收尾
     - assistant_playback_extra_chunks：助手 TTS 播完后多推的静音块数
   过短可能导致 segment_final 偏晚或 ASR 截断，请按日志与 _还原.txt 微调。

6) 客户主动打断（barge_in）自测（WAV 回放模拟「正在播」时用户开说）::

     .venv\Scripts\python.exe examples\full_duplex_simulate_from_wav.py --demo_barge_in --device xpu --chunk_ms 80

   急躁客户（默认完整用例：首轮 a01 说明 → u02 抢话 → 停播 → 间隔 → 二轮 a02 重讲；默认用户句 u02.wav）::

     .venv\Scripts\python.exe examples\full_duplex_simulate_from_wav.py --impatient_barge_in --device xpu --chunk_ms 80 --lead_in_silence_ms 2000

   仅「抢话」无第二轮讲解（旧版三阶段）::

     同上命令加 --impatient-no-retry

   二轮助手 / 间隔 / 客户听完后的短句可改::

     --assistant-wav（首轮参考，默认 a01.wav）  --assistant-wav-retry（默认 a02.wav）
     --post-interrupt-gap-ms   --customer-ack-wav（如 wavs\u03.wav，可选）

   默认可写入 <仓库根>/output/call_recordings（完整用例文件名含 impatient_retry ；仅抢话为 impatient_* ；与 cwd 无关）；关闭录音加 --no-call-audio。
   立体声文件：L=客户上行；R=与同一时间轴对齐的助手参考。完整用例下 R=首轮 a01（与 lead+抢话重叠）+ 停播留白 + 二轮 a02；仅 --impatient-no-retry 时 R≈单条 a01。
   可改 --assistant-wav / --lead_in_silence_ms；右轨静音调试用 --no-assistant-in-stereo。
   跑完终端打印绝对路径，并写同前缀 _call_audio.json（含 duplex_sim_scenario 摘要）。
   其它模式需录音时加 --call-audio；目录默认 output/call_recordings（勿与 output/logs 混用，以免清日志误删录音）。
   可改 --wav_path / --lead_in_silence_ms / --post_user_tail_ms。勿与 --demo_barge_in 同用。
   输出 JSON 行中应出现 event=barge_in（外放时 TTS 漏进麦克风易误报，建议耳机或已 AEC 的麦）。
   实麦 + LLM + TTS 全双工打断：examples\full_duplex_voice_llm_tts.py（播报中直接说话即可触发停止播放）。
