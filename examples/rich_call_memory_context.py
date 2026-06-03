"""富通话仿真：通话记忆与动态话务任务拼装（参考 hello-agents chapter9 上下文工程思路）。

- **追踪层**：按时间记录「客户·识别」批次与客服已播要点，拼成 ``【通话记忆·追踪】`` 前言块。
- **记忆智能体（可选）**：额外一次 LLM 调用，把追踪稿压成短摘要 ``【通话记忆·智能体摘要】``，减轻主模型在长对话上的负担。

场景 JSON 中可省略冗长 ``llm_user``，默认按轮次 ``name`` 查 `ROUND_TASK_CONSTRAINTS`；
若某轮仍提供 ``llm_user``，则**覆盖**注册表条目。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def _clip(s: str, n: int) -> str:
    t = (s or "").strip().replace("\n", " ")
    if len(t) <= n:
        return t
    return t[: n - 1] + "…"


# 与 duplex_rich_call_scenario/scenario.json 各 listen/barge 轮 name 对齐
ROUND_TASK_CONSTRAINTS: dict[str, str] = {
    "opening_verify": (
        "请输出外呼开场：像真人打电话一样短、脆。先称呼+问是不是张先生本人或是否本人接听，再自报移动幺零零八六客服。"
        "总共不超过两句、不超过40字。不要用「方便核实一下您的身份」等书面语；"
        "可用「麻烦您确认下是您本人吗」「请问是张先生本人吗」这类口头说法。不要一口气念「您好张先生我是…」。"
    ),
    "bridge_to_biz": (
        "结合「通话记忆」与**最新**「客户·识别」。外呼：勿说「感谢您的来电」。先简短致谢接听，再用一句确认本次主要是否咨询**套餐变更**，"
        "可顺带问是否还关心**话费余额**或**上网是否正常**（不超过12字带过即可）。总共不超过32字，不重复核身。"
    ),
    "biz_long": (
        "结合「通话记忆」与**最新**「客户·识别」（以识别稿为准；听不清勿编造原话）。你要**主动开讲**，勿用「您先说」等把话头完全交回。"
        "结构：①不超过16字承接；②用约70到90字说明：**套餐变更**可走移动App或幺零零八六、费用与生效以办理时为准；"
        "若识别稿涉及**话费/余额**，可穿插**一句演示性模拟查询**（合理区间或示例金额、每次可略有不同），并立即点明**以App与系统为准、本通为演示口径**；"
        "若涉及**上网慢、无信号**，夹带一句**初排**（重启、飞行模式、欠费、App 公告）；"
        "③收尾一句引导明细自查。全文不超过115字，禁止 markdown。"
    ),
    "retry_after_c1": (
        "结合「通话记忆」「客户·识别」与抢话内容。①不超过12字承接；②用约50到75字：资费/账单原则；"
        "若客户追问金额，可给**一句自洽的模拟查询表述**（数字勿夸张），并强调**非最终凭证、以中国移动 App 或幺零零八六核实**；"
        "若涉及信号/上网，补半句初排。总字数不超过92字。"
    ),
    "retry_after_c2": (
        "结合「通话记忆」「客户·识别」与抢话。①短承接；②约45到70字：合约/违约金以协议与系统为准，数字给不出就明说请幺零零八六或营业厅；"
        "若客户**不满或要投诉**，平静告知可通过**App 投诉评价**、**继续拨打幺零零八六升级受理**、以及按规定的**申诉渠道**反映，不争辩、不承诺赔偿。"
        "总字数不超过95字。"
    ),
    "goodbye_close": (
        "阅读**最新**「客户·识别」与下方「通话记忆」："
        "若客户已主动表示要挂电话、先这样、没事了、再见/拜拜等**结束通话**意向，你只作外呼极简道别（感谢配合、祝愉快等），"
        "一两句、不超过28字，**禁止**「感谢您的来电」，勿再追问业务、勿延长时间。"
        "**若**最新识别稿里看不出结束意向（极罕见），再用不超过22字简短确认一句是否还需要协助。"
    ),
}


@dataclass
class RichCallMemoryAgent:
    """维护通话侧事实追踪，并可选调用 LLM 生成短摘要（记忆智能体）。"""

    customer_batches: list[str] = field(default_factory=list)
    assistant_utterances: list[str] = field(default_factory=list)
    agent_summary: str = ""

    def add_customer_asr(self, text: str) -> None:
        t = (text or "").strip()
        if t:
            self.customer_batches.append(t)

    def add_assistant_spoken(self, text: str) -> None:
        t = (text or "").strip()
        if t:
            self.assistant_utterances.append(t)

    def clear_summary(self) -> None:
        self.agent_summary = ""

    def render_tracking_block(
        self,
        *,
        max_customer_batches: int = 4,
        max_assistant: int = 3,
        max_line_chars: int = 72,
    ) -> str:
        lines: list[str] = []
        cs = self.customer_batches[-max_customer_batches:]
        for i, c in enumerate(cs, start=1):
            lines.append(f"- 客户（第{len(self.customer_batches) - len(cs) + i}段）：{_clip(c, max_line_chars)}")
        aus = self.assistant_utterances[-max_assistant:]
        for i, a in enumerate(aus, start=1):
            lines.append(f"- 客服已播（第{len(self.assistant_utterances) - len(aus) + i}段）：{_clip(a, max_line_chars)}")
        if not lines:
            return "【通话记忆·追踪】\n（尚无已记录的客户识别批次或客服播报摘要。）"
        return "【通话记忆·追踪】\n" + "\n".join(lines)

    def render_preamble_for_task(self) -> str:
        """供拼进「话务任务」最前部：优先展示智能体摘要，否则展示追踪列表。"""
        if (self.agent_summary or "").strip():
            return "【通话记忆·智能体摘要】\n" + _clip(self.agent_summary, 360)
        return self.render_tracking_block()

    def raw_timeline_for_summarizer(self, *, max_chars: int = 1600) -> str:
        parts: list[str] = []
        for c in self.customer_batches:
            parts.append(f"客户：{c}")
        for a in self.assistant_utterances:
            parts.append(f"客服：{a}")
        return _clip("\n".join(parts), max_chars)

    def refresh_summary_with_llm(
        self,
        *,
        vm: Any,
        api_key: str,
        base_url: str,
        model: str,
        timeout_s: float,
        request_trace: list[dict[str, Any]] | None = None,
        trace_max_content_chars: int = 0,
    ) -> None:
        """记忆智能体：单次 LLM 压缩时间线（无外部 hello_agents 依赖）。"""
        timeline = self.raw_timeline_for_summarizer()
        if not timeline.strip():
            self.agent_summary = ""
            return
        sys_m = (
            "你是移动幺零零八六外呼场景的「记忆智能体」，只根据给定对话片段输出事实性摘要，供主客服模型生成下一句话术。"
            "可概括：客户身份线索、套餐/话费/网络/投诉相关意图、客服已交代要点、是否表露不满或要结束。"
            "禁止编造；听不清就写「听不清」。输出4～7条短句，每条不超过40字；不要编号以外的 markdown；不要复述编导指令。"
        )
        usr = (
            "下列为截至目前客户侧识别稿与客服已播内容（可能漏字、叠音）：\n"
            f"{timeline}\n\n"
            "请概括：客户身份线索、当前业务意图、客服已交代要点、客户离题或插话主题、是否表露要结束通话。"
        )
        try:
            out = vm.openai_compatible_chat(
                [
                    {"role": "system", "content": sys_m},
                    {"role": "user", "content": usr},
                ],
                api_key=api_key,
                base_url=base_url,
                model=str(model),
                timeout_s=float(timeout_s),
                request_trace=request_trace,
                trace_tag="memory_agent_summary",
                trace_max_content_chars=int(trace_max_content_chars),
                trace_extra={"role": "memory_agent"},
            )
        except Exception:
            self.agent_summary = ""
            return
        self.agent_summary = _clip(re.sub(r"\s+", " ", (out or "").strip()), 360)


def resolve_round_task_body(rd: dict[str, Any]) -> str:
    """轮次编导约束：JSON llm_user 优先，否则按 name 查注册表。"""
    u = str(rd.get("llm_user") or "").strip()
    if u:
        return u
    nm = str(rd.get("name") or "").strip()
    body = ROUND_TASK_CONSTRAINTS.get(nm, "")
    if not body:
        return ""
    return body


def compose_task_user_content(
    rd: dict[str, Any],
    memory: RichCallMemoryAgent,
    *,
    use_memory_preamble: bool = True,
) -> str:
    """拼装送入主 LLM 的 user 文本（再由 _format_agent_task_message 加【话务任务】）。"""
    task = resolve_round_task_body(rd)
    if not task:
        return ""
    if not use_memory_preamble:
        return task
    pre = memory.render_preamble_for_task().strip()
    if not pre:
        return task
    return f"{pre}\n\n【本轮编导约束】\n{task}"
