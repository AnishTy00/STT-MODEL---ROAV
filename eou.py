"""Transcript-aware End-of-Utterance (EOU) detection."""

from __future__ import annotations

# Punctuation marks indicating sentence completion
SENTENCE_END_PUNCTUATION = (".", "?", "!", "...")

# Words suggesting the speaker is about to continue
CONTINUATION_WORDS = {
    "and", "but", "or", "because", "then", "and then", "if", "so", 
    "such as", "like", "although", "while", "to", "with", "for", 
    "about", "because of", "i mean", "you know"
}


# Known Whisper hallucination phrases when transcribing silence or noise
HALLUCINATION_PHRASES = {
    "thank you", "thanks for watching", "thanks for listening", "subscribe",
    "subtitles by", "subtitle by", "music", "[music]", "(music)", "you",
    "i'm sorry", "i am sorry", "i'm going to", "i am going to", "thank you very much",
    "she was", "he was", "it was", "bye-bye"
}


def is_hallucination(text: str) -> bool:
    """Check if the text is a known hallucination phrase."""
    normalized = " ".join(text.lower().strip(" .,!?:;[]()").split())
    return normalized in HALLUCINATION_PHRASES


def is_sentence_complete(text: str) -> bool:
    """Check if the text ends with sentence-ending punctuation."""
    trimmed = text.strip()
    if not trimmed:
        return False
    return trimmed.endswith(SENTENCE_END_PUNCTUATION)


def ends_with_continuation(text: str) -> bool:
    """Check if the text ends with a continuation word/phrase."""
    # Normalize text by converting to lowercase and stripping punctuation/extra spacing
    normalized = text.lower().strip(" .,!?:;[]()")
    if not normalized:
        return False

    # Extract the last one or two words to check for multi-word continuations (e.g. "and then")
    words = normalized.split()
    if not words:
        return False

    last_word = words[-1]
    last_two_words = " ".join(words[-2:]) if len(words) >= 2 else ""

    return last_word in CONTINUATION_WORDS or last_two_words in CONTINUATION_WORDS


def evaluate_eou(
    text: str,
    silence_seconds: float,
    eou_short_silence_seconds: float = 0.5,
    end_silence_seconds: float = 1.5,
    enable_transcript_eou: bool = True,
) -> bool:
    """Layered EOU logic to determine if the speaker has finished.
    
    Returns True if EOU is detected, False otherwise.
    """
    # 1. Unconditional silence timeout: fallback safety net
    if silence_seconds >= end_silence_seconds:
        return True

    # 2. Too short pause: never trigger EOU
    if silence_seconds < eou_short_silence_seconds:
        return False

    # 3. If transcript-aware EOU is disabled, we rely solely on standard silence timeout
    if not enable_transcript_eou:
        return False

    trimmed = text.strip()
    if not trimmed:
        return False

    # 4. Completed sentence punctuation: Whisper naturally inserts period/question/exclamation
    # marks when a thought/sentence is grammatically complete.
    if is_sentence_complete(trimmed):
        return True

    # 5. Continuation phrases: if the speaker ended with "and", "because", etc.
    # we definitely do NOT want to end the utterance early.
    if ends_with_continuation(trimmed):
        return False

    # 6. Question words or command prefixes often signal complete thoughts even without explicit punctuation.
    # If the pause is at least 70% of the maximum silence threshold, and there is no continuation, 
    # we can finalize to keep it responsive.
    midway_silence = eou_short_silence_seconds + 0.7 * (end_silence_seconds - eou_short_silence_seconds)
    if silence_seconds >= midway_silence:
        return True

    return False
