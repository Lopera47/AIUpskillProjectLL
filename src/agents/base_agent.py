"""Base class for AI agents."""
from abc import ABC, abstractmethod
from typing import Dict, Any
import google.genai as genai
import os


class BaseAgent(ABC):
    """
    Base class for all AI agents.

    Implements Template Method pattern:
    - execute() orchestrates the workflow
    - Subclasses implement specific steps
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize agent.

        Args:
            model_name: Gemini model to use
        """
        self.model_name = model_name
        self.client = None
        self._configure_client()

    def _configure_client(self):
        """Configure Gemini client."""
        # Get API key from environment
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")

        # Create client with API key
        self.client = genai.Client(api_key=api_key)

    async def execute(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Execute agent workflow (Template Method).

        Steps:
        1. Load input context
        2. Process with LLM
        3. Save results

        Args:
            input_path: Path to input markdown file
            output_path: Path to save output

        Returns:
            Result metadata
        """
        print(f"\n🤖 {self.__class__.__name__} starting...")

        # Step 1: Load
        context = await self._load_context(input_path)

        # Step 2: Process
        result = await self._process(context)

        # Step 3: Save
        await self._save_result(result, output_path)

        print(f"✅ {self.__class__.__name__} complete")

        return {
            'input_path': input_path,
            'output_path': output_path,
            'success': True
        }

    @abstractmethod
    async def _load_context(self, input_path: str) -> Dict[str, Any]:
        """
        Load input context.

        Subclasses implement how to read their input.
        """
        pass

    @abstractmethod
    async def _process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process context with LLM.

        Subclasses implement their specific logic.
        """
        pass

    @abstractmethod
    async def _save_result(self, result: Dict[str, Any], output_path: str):
        """
        Save processing result.

        Subclasses implement how to save their output.
        """
        pass

    def _call_llm(self, prompt: str) -> str:
        """
        Call Gemini with prompt.

        Helper method for subclasses.

        Args:
            prompt: Prompt to send
            system_instruction: Optional system prompt

        Returns:
            LLM response text
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"❌ LLM call failed: {e}")
            raise