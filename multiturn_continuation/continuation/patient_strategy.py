"""
Patient Strategy Module

Handles patient response generation using DirectPatient LLM.

"""

import sys
import os
from typing import List, Dict, Optional

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


def convert_to_patient_format(conversation_history: List[Dict]) -> List[Dict]:
    """
    Convert from role-based format to Patient agent format.

    Input: [{"role": "Patient", "content": "...", "turn_index": N}, ...]
    Output: [{"Patient": "...", "turn index": N}, {"Doctor": "...", "turn index": M}, ...]
    """
    patient_format = []
    for turn in conversation_history:
        role = turn.get('role', '')
        content = turn.get('content', '')
        turn_index = turn.get('turn_index', 0)

        if role == 'Patient':
            patient_format.append({'Patient': content, 'turn index': turn_index})
        elif role == 'Doctor':
            patient_format.append({'Doctor': content, 'turn index': turn_index})

    return patient_format


class PatientContinuationStrategy:
    """Generates patient responses using a DirectPatient LLM agent."""

    def __init__(self, patient_agent=None):
        self.patient_agent = patient_agent

    def get_next_patient_response(
        self,
        case: Dict,
        conversation_history: List[Dict],
    ) -> Optional[Dict]:
        """
        Generate next patient response using the LLM agent.

        Returns:
            Patient turn dict or None if agent unavailable / missing data.
        """
        if not self.patient_agent:
            print("⚠️  Warning: No patient agent available")
            return None

        if not case.get('complete_conversation'):
            print("⚠️  Warning: No complete_conversation for generation")
            return None

        truncated_conv = convert_to_patient_format(conversation_history)
        complete_conv = case['complete_conversation']

        try:
            response_dict = self.patient_agent.respond(
                truncated_conversation=truncated_conv,
                complete_conversation=complete_conv
            )

            patient_response = response_dict.get('response', '')
            last_turn_index = conversation_history[-1]['turn_index'] if conversation_history else 0

            return {
                'role': 'Patient',
                'content': patient_response,
                'turn_index': last_turn_index + 1,
                'source': 'llm_generated'
            }

        except Exception as e:
            print(f"❌ Error generating patient response: {e}")
            return None


if __name__ == "__main__":
    print("Testing Patient Strategy Module...")

    role_based = [
        {"role": "Patient", "content": "Hello", "turn_index": 0},
        {"role": "Doctor", "content": "Hi", "turn_index": 1}
    ]

    patient_format = convert_to_patient_format(role_based)
    print(f"\nFormat conversion test:")
    print(f"  Input: {role_based}")
    print(f"  Output: {patient_format}")
