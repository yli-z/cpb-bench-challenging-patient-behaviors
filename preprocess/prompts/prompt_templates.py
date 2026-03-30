"""
Prompt templates for behavior detection (Chinese and English).
"""

# English user prompt template
EN_USER_PROMPT_TEMPLATE = """Analyze the following conversations and identify all patient behaviors.

## Instructions:
- Analyze each conversation below (includes both doctor and patient turns)
- Look for behaviors across multiple turns, not just single utterances
- Flag any turn that exhibits one or more behaviors (multi-label allowed)
- Use confidence scores: 0.9+ (obvious and clearly meets conditions), 0.7-0.9 (likely meets conditions), 0.5-0.7 (possibly meets conditions but uncertain)
- Always provide the full conversation as conversation_segment

**For multi-turn behaviors (e.g., Information Contradiction that spans multiple turns):**
- The "turn_index" should be the uttr_id of the PRIMARY turn (usually the later turn where the contradiction becomes evident)
- The "related_turn_indices" should list ALL related turn_indices (e.g., [3, 7] for a contradiction between turn 3 and turn 7)
- The "patient_text" should contain ONLY the patient's text from the primary turn_index

**For single-turn behaviors:**
- The "turn_index" should be the uttr_id of the patient utterance where the behavior is detected
- The "related_turn_indices" should be an empty array []
- The "patient_text" should contain the patient's utterance text at that turn_index

## Conversations to Analyze:

{conversations}

Analyze all conversations above and return all patient behaviors found. If no behaviors are detected, return an empty array []."""

# Chinese user prompt template
ZH_USER_PROMPT_TEMPLATE = """分析以下对话并识别所有患者行为。

## 分析指导：

1. 分析下方提供的每一段对话（包含医生和患者的全部轮次）
2. 在多轮对话中寻找行为模式，而不仅限于单次发言
3. 每轮对话允许存在一种或多种行为的患者轮次（允许多标签），请全部标出
4. 使用置信度评分：0.9+（明显且完全符合条件），0.7-0.9（很可能符合条件），0.5-0.7（可能符合条件但不确定）
5. 始终提供完整对话作为 conversation_segment

**针对跨多轮次的行为（例如，跨越多个轮次的信息矛盾）：**
- "turn_index" 应填写主要轮次的 uttr_id（通常是矛盾变得明显的后一轮次）
- "related_turn_indices" 应列出所有相关的轮次索引（例如，若矛盾发生在第3轮和第7轮，则填写 [3,7]）
- "patient_text" 应仅包含主要轮次中患者的文本（不要包含多个轮次的文本）

**针对单轮次内出现的行为：**
- "turn_index" 应填写检测到行为的患者发言对应的 uttr_id
- "related_turn_indices" 应为空数组 []
- "patient_text" 应包含该轮次中患者的发言文本

## 待分析的对话：

{conversations}

分析上述所有对话，返回找到的所有患者行为。如果未检测到行为，返回空数组 []。"""


def get_user_prompt_template(language: str = "en") -> str:
    """Get user prompt template based on language.
    
    Args:
        language: "en" for English, "zh" for Chinese
        
    Returns:
        User prompt template string
    """
    if language.lower() == "zh":
        return ZH_USER_PROMPT_TEMPLATE
    return EN_USER_PROMPT_TEMPLATE

