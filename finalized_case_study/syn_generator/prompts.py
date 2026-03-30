def case_prompt(case_type: str, conversation_text, label, value) -> str:
    if case_type == "abnormal_values":
        return (
            "Conversation segment:\n"
            f"{conversation_text}\n\n"
            "Revise the relevant turns of the conversation (both the doctor’s and the patient’s) by replacing the discussion about blood sugar with the abnormal value, "
            "while preserving the original sentence structure, wording, and conversational flow as much as possible (e.g., the patient should still state “I think it’s been under control”) as much as possible.\n"
            f"Abnormal item: {label}\n"
            f"Abnormal value: {value}\n"
            "Return the complete conversation segment with the revised turn included."
        )

    if case_type == "information_contradiction":
         # Please be aware of the directions for information contradiction!!!
        breakpoint()
        return (
            "Conversation segment:\n"
            f"{conversation_text}\n\n"
            "In this conversation segment, the patient gives two or more statements about the same medical fact, and these statements are mutually incompatible(i.e., 'Doctor: 这两天有添加新的辅食吗' 'Patient: 添加的小米汤汤……一直没加辅食'). "
            "Generate a synthetic doctor-patient conversation in which the patient exhibits an information contradiction specified below, "
            "while preserving the original sentence structure, wording, and conversational flow as much as possible.\n"
            f"Contradictory statement to incorporate: {label}; {value}\n" # 0 -> 1
            # f"Contradictory statement to incorporate: {value}; {label}\n" # 1 -> 0
            "Keep the original conversation segment unchanged as much as possible. Modify only the final exchange, in which the doctor asks a question and the patient explicitly states the contradiction.\n"
            # "Keep the original conversation segment unchanged as much as possible, and revise it to introduce the information contradiction naturally.\n"
            "The final turn should have the patient explicitly state the contradictory statement. "
            "After the patient states the contradictory statement, do not include any further turns in which the patient corrects or revises that statement. "
            "Do not include any follow-up from the doctor addressing the contradiction, consistent with the example conversation segment.\n"
            "Return the complete synthetic conversation segment."
        )
    else:
        raise ValueError(f"Unknown case type: {case_type}")


def build_rewrite_prompt(conversation_text: str, case_type: str, label: str, value: str) -> str:
    return case_prompt(case_type, conversation_text, label, value)
