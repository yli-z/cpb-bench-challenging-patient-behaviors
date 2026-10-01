"""
Multi-turn Continuation Engine

Core engine for generating multi-turn dialogue continuations.
This module ONLY handles dialogue generation, NOT evaluation.
"""

import sys
import os
from typing import List, Dict

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from continuation.patient_strategy import PatientContinuationStrategy



class MultiTurnContinuationEngine:
    """
    Core engine for multi-turn dialogue generation.
    This engine ONLY generates dialogues, does NOT perform evaluation.
    """
    
    def __init__(self, doctor_agent, patient_agent=None, verbose=True):
        """
        Args:
            doctor_agent: Doctor agent (DirectDoctor or similar)
            patient_agent: Optional Patient agent (DirectPatient) for generation mode
            verbose: Print progress
        """
        self.doctor_agent = doctor_agent
        self.patient_strategy = PatientContinuationStrategy(patient_agent)
        self.verbose = verbose
    
    def _print(self, message: str):
        """Print if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def run_continuation(
        self,
        case: Dict,
        max_turns: int = 10,
        max_retries: int = 3
    ) -> Dict:
        """
        Generate multi-turn dialogue continuation for a single case.
        This method ONLY generates dialogues, does NOT perform evaluation.

        Args:
            case: Failed case data
            max_turns: Maximum number of additional turns to generate
            max_retries: Maximum retries if no doctor turn is generated

        Returns:
            Result dictionary with generated dialogue:
            {
                'case_id': str,
                'behavior_category': str,
                'multi_turn_response': List[Dict],  # Newly generated turns
                'conversation_history': List[Dict]  # Full conversation (original + new)
            }
        """
        case_id = case.get('case_id', 'unknown')
        behavior_category = case.get('behavior_category', 'unknown')

        self._print(f"\n{'='*80}")
        self._print(f"🔄 Generating continuation for: {case_id}")
        self._print(f"   Category: {behavior_category}")
        self._print(f"   Max turns: {max_turns}")
        self._print(f"{'='*80}")

        for attempt in range(1, max_retries + 1):
            if attempt > 1:
                self._print(f"\n⚠️  Retry {attempt}/{max_retries} (no doctor turn generated in previous attempt)")

            # Initialize (reset on each attempt)
            conversation_history = case['conversation_history'].copy()
            multi_turn_response = []  # Store newly generated turns
            rounds_completed = 0

            # Track the starting point (length of original conversation)
            original_length = len(conversation_history)

            # Run continuation loop
            for round_num in range(max_turns):
                self._print(f"\n--- Round {round_num + 1}/{max_turns} ---")

                # Step 1: Get Patient Response
                patient_turn = self.patient_strategy.get_next_patient_response(
                    case=case,
                    conversation_history=conversation_history,
                )

                if not patient_turn:
                    self._print("   ⚠️  No more patient turns available. Stopping.")
                    break

                patient_response = patient_turn.get('content', '')
                patient_turn_index = patient_turn.get('turn_index', 0)
                source = patient_turn.get('source', 'llm_generated')

                self._print(f"   Patient ({source}): {patient_response[:80]}...")

                # Check if patient wants to end conversation
                if '[End of Conversation]' in patient_response:
                    self._print("   ✅ Patient indicated end of conversation. Stopping.")
                    # Still add the patient turn but don't generate doctor response
                    conversation_history.append(patient_turn)
                    multi_turn_response.append(patient_turn)
                    break

                # Add to history
                conversation_history.append(patient_turn)

                # Step 2: Doctor Responds
                try:
                    doctor_response, doctor_thinking = self.doctor_agent.respond(
                        current_conversation=conversation_history,
                        complete_conversation=case.get('complete_conversation', [])
                    )

                except Exception as e:
                    self._print(f"   ❌ Doctor generation error: {e}")
                    break

                doctor_turn_index = patient_turn_index + 1

                self._print(f"   Doctor: {doctor_response[:80]}...")

                # Create doctor turn
                doctor_turn = {
                    'role': 'Doctor',
                    'content': doctor_response,
                    'turn_index': doctor_turn_index,
                    'source': 'llm_generated',
                    'round': round_num + 1
                }

                # Save thinking content if available (e.g., Qwen3 thinking mode)
                if doctor_thinking:
                    doctor_turn['thinking'] = doctor_thinking

                # Add to history
                conversation_history.append(doctor_turn)

                # Add both patient and doctor turns to multi_turn_response
                multi_turn_response.append(patient_turn)
                multi_turn_response.append(doctor_turn)

                rounds_completed += 1

            # Count doctor turns generated in this attempt
            doctor_turn_count = sum(
                1 for t in multi_turn_response
                if t.get('role') == 'Doctor' and t.get('source') == 'llm_generated'
            )
            self._print(f"\n   Doctor turns generated: {doctor_turn_count}")

            if doctor_turn_count >= 1:
                break  # Sufficient turns generated

            if attempt < max_retries:
                self._print(f"   ⚠️  Doctor turn count ({doctor_turn_count}) < 1. Retrying...")
            else:
                self._print(f"   ⚠️  Doctor turn count still < 1 after {max_retries} attempts. Proceeding anyway.")

        # Build the final multi_turn_response.
        # Prepend the three turns that provide full context:
        # [doctor_original_question, patient_behavior (original), llm_failed_doctor_response].
        # These are the last three entries of the *original* conversation_history
        # (before any generation appended new turns).
        if original_length >= 3:
            prefix_turns = list(conversation_history[original_length - 3 : original_length])
        elif original_length >= 2:
            prefix_turns = list(conversation_history[original_length - 2 : original_length])
        elif original_length == 1:
            prefix_turns = list(conversation_history[:1])
        else:
            prefix_turns = []

        full_multi_turn_response = prefix_turns + multi_turn_response

        self._print(f"\n{'='*80}")
        self._print(f"✅ Continuation generation complete!")
        self._print(f"   Rounds: {rounds_completed}")
        self._print(f"   New turns generated: {len(multi_turn_response)}")
        self._print(f"   Total multi_turn_response (incl. prefix): {len(full_multi_turn_response)}")
        self._print(f"   Total conversation length: {len(conversation_history)}")
        self._print(f"{'='*80}")

        return {
            'case_id': case_id,
            'behavior_category': behavior_category,
            'model': case.get('model', 'unknown'),
            'multi_turn_response': full_multi_turn_response,
            'conversation_history': conversation_history
        }


if __name__ == "__main__":
    print("Multi-turn Continuation Engine module loaded successfully.")
