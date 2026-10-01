def format_conversations_to_string(conversations: list, is_complete: bool) -> str:
    conversation_text = ""
    for talk in conversations:
        if "Doctor" in talk:
            role = "Doctor"
        elif "Patient" in talk:
            role = "Patient"
        elif "control" in talk:
            continue
        else:
            raise ValueError(f"Unknown role in talk: {talk}")

        content = talk[role]
        conversation_text += f"{role}: {content}\n"

    conversation_text = f"{'Complete' if is_complete else 'Truncated'} Conversation:\n{conversation_text}"
    return conversation_text


def format_conversations_role_content_to_string(conversation_history):
    conversation_str = ""
    for talk in conversation_history:
        role = talk.get('role', '')
        content = talk.get('content', '')
        role_lower = role.lower()
        if role_lower == 'doctor':
            display_role = "Doctor"
        elif role_lower == 'patient':
            display_role = "Patient"
        elif role_lower == 'system':
            continue
        else:
            display_role = role.capitalize()
        conversation_str += f"{display_role}: {content}\n"
    return conversation_str


def format_json_to_string(data: dict) -> str:
    out_str = ""
    for key, value in data.items():
        out_str += f"{key}: {value}\n"
    return out_str


def format_prompt_to_messages(user_prompt: str, system_prompt: str = None) -> list:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages
