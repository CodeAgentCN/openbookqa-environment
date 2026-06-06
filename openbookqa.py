import re
from typing import Optional

import verifiers as vf
from datasets import Dataset, load_dataset
from verifiers.parsers.parser import Parser
from verifiers.types import Messages

LETTERS = ("A", "B", "C", "D")

class OpenBookQAParser(Parser):
    """Parser for OpenBookQA multiple choice answers."""
    
    _BOXED = re.compile(r"\\boxed\{([ABCD])\}", re.IGNORECASE)
    _LABELED = re.compile(r"(?:ANSWER|CHOICE|SELECT|PICK)\s*(?:IS|[:=])?\s*\(?([ABCD])\b", re.IGNORECASE)
    
    def parse(self, text: str) -> Optional[str]:
        if not text:
            return None
        text = text.strip().upper()
        
        if text in LETTERS:
            return text
        
        if m := self._BOXED.search(text):
            return m.group(1)
        
        if m := self._LABELED.search(text):
            return m.group(1)
        
        # Look for first A-D in text
        for letter in LETTERS:
            if letter in text:
                return letter
        return None
    
    def parse_answer(self, completion: Messages) -> Optional[str]:
        content = completion[-1]["content"] if isinstance(completion, list) else completion
        return self.parse(content)


def load_environment(split: str = "test", **kwargs) -> vf.Environment:
    """Load OpenBookQA science Q&A environment.
    
    Args:
        split: Dataset split - "train", "validation", or "test"
    """
    valid_splits = ["train", "validation", "test"]
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}")
    
    def generator():
        raw = load_dataset("openbookqa", "main", split=split)
        
        for ex in raw:
            question = ex["question_stem"]
            choices = ex["choices"]
            answer = ex["answerKey"]  # A, B, C, or D
            
            # Format choices
            choice_text = ""
            for i, (label, text) in enumerate(zip(choices["label"], choices["text"])):
                choice_text += f"{label}) {text}\n"
            
            prompt = f"Question: {question}\n\nChoices:\n{choice_text}\n\nSelect the correct answer."
            
            yield {
                "messages": [{"role": "user", "content": prompt}],
                "ground_truth": answer,
                "metadata": {"source": "openbookqa", "split": split}
            }
    
    return vf.SingleTurnEnv(
        dataset=Dataset.from_generator(generator),
        parser=OpenBookQAParser(),
    )
