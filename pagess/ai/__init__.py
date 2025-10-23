

try:
    from .ai_assistant import render_ai_assistant
except ImportError as e:
    print(f"Warning: Could not import ai_assistant: {e}")
    render_ai_assistant = None


# Définir ce qui est exporté
__all__ = [
   
    'render_ai_assistant'
   
]
