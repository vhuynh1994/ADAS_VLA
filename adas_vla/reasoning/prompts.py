"""Prompt templates. Shared by inference, auto-labeling and fine-tuning so they never drift apart."""

from __future__ import annotations

from ..types import LatAction, LongAction

LANGUAGE_NAMES = {"en": "English", "vi": "Vietnamese"}

# Listed with the default first (models favour the first option) and a short definition each.
LONG_ACTION_HELP = {
    LongAction.KEEP: "hold current speed or gap",
    LongAction.ACCELERATE: "speed up, only if below cruise speed and the lane ahead is clear",
    LongAction.DECELERATE: "ease off to open the gap or slow for a hazard",
    LongAction.BRAKE: "firm braking for a close or sudden hazard",
    LongAction.STOP: "come to a full stop (red light, stop sign, blocked road)",
}
LAT_ACTION_HELP = {
    LatAction.KEEP_LANE: "stay centered in the lane",
    LatAction.NUDGE_LEFT: "small offset left inside the lane",
    LatAction.NUDGE_RIGHT: "small offset right inside the lane",
    LatAction.CHANGE_LEFT: "move to the left lane",
    LatAction.CHANGE_RIGHT: "move to the right lane",
}
VLM_LONG_ACTIONS = [a.value for a in LONG_ACTION_HELP]
VLM_LAT_ACTIONS = [a.value for a in LAT_ACTION_HELP]


def _action_menu(help_map: dict) -> str:
    return "; ".join(f"{a.value} = {text}" for a, text in help_map.items())

# The assistant turn is pre-filled with this so the output always starts with the decision. It ends right
# before the action value, so the next token alone identifies the longitudinal action (KEEP | AC | DE | B |
# STOP): its probabilities are read for free during generation (see vlm.ActionPolicy).
DECISION_PREFIX = '{"longitudinal": "'

DECISION_SYSTEM_PROMPT = (
    "You are the decision module of an Advanced Driver Assistance System (ADAS). "
    "You see the front camera image of the ego vehicle and structured perception data. "
    "Choose one longitudinal action (speed) and one lateral action (lane). Keep a safe following "
    "distance, yield to pedestrians and cyclists, obey traffic lights and signs, and keep the lane "
    "unless there is a clear reason to leave it. Oncoming traffic on the other side of a median is "
    "not a hazard. Respond with one compact JSON object only."
)

DECISION_USER_TEMPLATE = """Ego speed: {ego_speed:.0f} km/h. Cruise set speed: {cruise_speed:.0f} km/h.
{context}

longitudinal: {long_actions}
lateral: {lat_actions}
Return JSON: {{"longitudinal": ..., "lateral": ..., "target_speed_kmh": <int>, "risk": "low|medium|high", {explain}"reason": "{reason_len}"}}{language_note}"""
# vlm.explain: the decision fields first (the action is still the first generated token), then a short scene
# description and the hazards that matter (scripts/explain_labels.py writes such targets with a teacher VLM)
EXPLAIN_FIELDS = '"scene": "<max 15 words>", "hazards": "<max 15 words>", '
EXPLAIN_STOP = '"scene"'  # decision-only decoding of an explaining VLM stops here (the reason comes last)

CHAT_SYSTEM_PROMPT = (
    "You are an in-car driving copilot. Answer the driver's questions about the current road "
    "scene using the camera image and the perception data. Be concise and put safety first. "
    "If you are not sure about something in the image, say so. Active safety alerts come from the "
    "validated ADAS system: never contradict them, and mention them first when they are relevant. "
    "Answer in {language}."
)


def alerts_text(alerts) -> str:
    """Safety alerts appended to the copilot context so the model cannot talk them away."""
    if not alerts:
        return "\nActive safety alerts: none."
    items = "; ".join(f"{a.kind} ({a.level}): {a.message}" for a in alerts)
    return f"\nActive safety alerts from the ADAS system: {items}."


def _language_note(language: str) -> str:
    if language == "en":
        return ""
    name = LANGUAGE_NAMES.get(language, language)
    return f"\nWrite the reason in {name}; keep keys and actions in English."


def visual_content(image) -> dict:
    """One image, or a list of frames (oldest first) sent as a short video (see VLMConfig.prev_frame_s)."""
    if isinstance(image, (list, tuple)):
        return {"type": "video", "video": list(image)}
    return {"type": "image", "image": image}


def decision_messages(image, context_text: str, ego_speed_kmh: float, cruise_speed_kmh: float,
                      language: str = "en", explain: bool = False) -> list[dict]:
    """Chat messages asking the VLM for a driving decision on one frame (or a [previous, current] pair);
    `explain` asks for the longer answer (scene, hazards, reason) of an explaining VLM."""
    user_text = DECISION_USER_TEMPLATE.format(
        ego_speed=ego_speed_kmh, cruise_speed=cruise_speed_kmh, context=context_text,
        long_actions=_action_menu(LONG_ACTION_HELP), lat_actions=_action_menu(LAT_ACTION_HELP),
        language_note=_language_note(language), explain=EXPLAIN_FIELDS if explain else "",
        reason_len="<max 20 words>" if explain else "<max 12 words>",
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": DECISION_SYSTEM_PROMPT}]},
        {"role": "user", "content": [visual_content(image), {"type": "text", "text": user_text}]},
    ]


def chat_messages(image, context_text: str, ego_speed_kmh: float, history: list[tuple[str, str]],
                  question: str, language: str = "en") -> list[dict]:
    """Multi-turn copilot chat. The image and perception summary are attached to the first turn."""
    system = CHAT_SYSTEM_PROMPT.format(language=LANGUAGE_NAMES.get(language, language))
    preamble = f"Ego speed: {ego_speed_kmh:.0f} km/h.\n{context_text}\n\n"
    messages = [{"role": "system", "content": [{"type": "text", "text": system}]}]
    turns = history + [(question, None)]
    for i, (q, a) in enumerate(turns):
        content = [{"type": "text", "text": (preamble + q) if i == 0 else q}]
        if i == 0:
            content.insert(0, {"type": "image", "image": image})
        messages.append({"role": "user", "content": content})
        if a is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": a}]})
    return messages


# Thinking mode (demo): OpenGVLab's InternVL3.5 thinking system prompt ("R1", model card); other VLMs follow it as a
# chain-of-thought instruction. The ADAS decision instructions move from the system turn to the user turn.
THINKING_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()  # noqa: E501


THINKING_INSTRUCTION = ("Before the JSON, think step by step inside <think> and </think>: what you see, which road users "
                        "could be hit and how soon, and which action is safest. Then output the JSON.")
THINKING_PREFIX = "<think>\n"  # pre-filled start of the reply: models without a native thinking mode think too


def thinking_messages(messages: list[dict]) -> list[dict]:
    """decision_messages() rewritten for thinking mode: the thinking protocol as system prompt."""
    system = messages[0]["content"][0]["text"]
    visual, text = messages[1]["content"]
    return [{"role": "system", "content": [{"type": "text", "text": THINKING_SYSTEM_PROMPT}]},
            {"role": "user", "content": [visual, {"type": "text", "text": system + "\n\n" + text["text"]
                                                  + "\n\n" + THINKING_INSTRUCTION}]}]


def split_thinking(text: str) -> tuple[str, str]:
    """(thinking, answer) of a thinking-mode reply; an unfinished <think> section leaves no answer."""
    start = text.find("<think>")
    if start == -1:
        return "", text
    end = text.find("</think>", start)
    if end == -1:
        return text[start + 7:].strip(), ""
    return text[start + 7:end].strip(), text[end + 8:]
