富通话场景 v6（贴近真实能力 + 随机客户音轨 + 女客服/男客户统一声线）
================================================================

剧情摘要
--------
1. 助手开场外呼核身（移动「幺零零八六」，不写阿拉伯数字 10086）。
2. **客户身份确认**：从 ``customer_pool.json`` 随机选一种说法（仅音轨）。
3. 助手一句过渡到套餐（可顺带带过话费/网络关切）。
4. 三轮 **barge**：客户侧为 **业务向抢话句** 拼接（``min_tangents``/``max_tangents`` 为 0 时不混入天气/球赛跑题），用于 **ASR/打断评测**；**不向大模型注入客户逐字稿**。``system_prompt`` 约定座席具备**话费演示查询**、**故障初排**、**投诉引导**（当前未接真系统，答复为模拟；后续可 MCP 真查）。
5. ``meta.random_customer_plan`` 中保留拼接原文，仅供事后对照 ASR，**非 LLM 输入**。
6. 助手在客户抢话场景下宜**先短句再接要点**，可说「不清楚」并引导 App/幺零零八六，忌长篇念稿（见 ``scenario.json`` v5）。
7. 助手 Edge **晓晓**（``EDGE_VOICE_ASSISTANT_FEMALE_DEFAULT``）；客户 Edge **云希**男声（与 prepare 脚本常量一致）。**客户侧**整批同引擎；auto 时先试 Edge，失败则**全部** pyttsx3，并在 SAPI 中固定**男声**匹配，避免与客服女声混淆。需全 Edge 时可 ``--engine edge``。

步骤
----
1) 从话术池生成全部客户 WAV（整批同一人声）::

     .venv\Scripts\python.exe scripts\prepare_rich_call_scenario_wavs.py --force

2) 回放（读 .env LLM）::

     .venv\Scripts\python.exe -X utf8 examples\full_duplex_rich_call_llm_sim.py --call-audio

   加 ``--call-audio`` 时：LLM+TTS 拼轴后**先写立体声 WAV**，再 ASR 推流。

3) 调助手语速：默认 ``+8%``；可 ``--assistant-edge-rate "-4%"`` 等。

4) 系统化矩阵（5 案例：4 个顺畅结案无抢话轮 + 1 个仅首轮礼貌抢话，见 ``scenario_matrix_*.json``）::

     .venv\Scripts\python.exe scripts\run_rich_call_scenario_matrix.py --skip-prepare --tts-engine edge
