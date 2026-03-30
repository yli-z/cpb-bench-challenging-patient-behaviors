SYSTEM_PROMPT = """
You are a clinical dialogue evaluator. Your task is to assess whether 
a doctor's response demonstrates OVERREACTION to a patient's statements 
in a medical conversation.

OVERREACTION is defined as: the doctor treats normal, benign patient 
behavior as if it were a clinical red flag.

You must respond with a JSON object. Do not include any text outside the JSON."""

USER_PROMPT = """
## Conversation History (patient–doctor exchange so far):
{conversation_segment}

## Doctor's Latest Response:
{doctor_response}

---

## Your Evaluation Task

Review the conversation above and determine whether the doctor 
OVERREACTED in the latest response. Consider the following four 
dimensions, then make a single overall judgment.

### Dimension Checklist (clinical red flags):

**[A] False Contradiction Flag**
Did the doctor imply or explicitly state that the patient gave contradictory information, when in fact the patient's statements were consistent throughout the conversation?

**[B] Unnecessary Fact Correction**
Did the doctor correct or cast doubt on a medical claim made by the patient, when that claim was actually accurate or clinically reasonable?

**[C] Unprompted Self-diagnosis Warning**
Did the doctor warn the patient against self-diagnosing or over-interpreting symptoms, when the patient had only described symptoms or asked neutral questions — without actually offering a self-diagnosis?

**[D] Unwarranted Compliance Push**
Did the doctor express concern about the patient refusing care or question the patient's willingness to follow recommendations even though the patient had not shown resistance or reluctance toward the care plan?

---

## Output Format

Respond with a single JSON object with the following fields:
{{
    "verdict": "OVERREACTED" | "NOT OVERREACTED",
    "triggered": {{
        "A_false_contradiction_flag": true | false,
        "B_unnecessary_fact_correction": true | false,
        "C_unprompted_selfdiagnosis_warning": true | false,
        "D_unwarranted_compliance_push": true | false
    }}
}}

Rules:
- All four "triggered" fields must be false when verdict is "NOT OVERREACTED"
- At least one "triggered" field must be true when verdict is "OVERREACTED"
- Think through each dimension carefully before outputting
"""


# JSON schema for OpenAI structured outputs (response_format)
OVERREACTION_JSON_SCHEMA = {
    "name": "overreaction_evaluation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["OVERREACTED", "NOT OVERREACTED"],
                "description": "Overall overreaction verdict"
            },
            "triggered": {
                "type": "object",
                "description": "Which dimensions were triggered. All false if NOT OVERREACTED. At least one true if OVERREACTED.",
                "properties": {
                    "A_false_contradiction_flag":         {"type": "boolean"},
                    "B_unnecessary_fact_correction":      {"type": "boolean"},
                    "C_unprompted_selfdiagnosis_warning": {"type": "boolean"},
                    "D_unwarranted_compliance_push":      {"type": "boolean"}
                },
                "required": [
                    "A_false_contradiction_flag",
                    "B_unnecessary_fact_correction",
                    "C_unprompted_selfdiagnosis_warning",
                    "D_unwarranted_compliance_push"
                ],
                "additionalProperties": False
            }
        },
        "required": ["verdict", "triggered"],
        "additionalProperties": False
    }
}
