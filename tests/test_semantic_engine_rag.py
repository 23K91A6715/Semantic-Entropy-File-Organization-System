import unittest
from unittest.mock import patch

from semantic_engine import SemanticEngine


class _FakePipeline:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompt = ""

    def __call__(self, prompt: str, **_kwargs: object) -> list[dict[str, str]]:
        self.prompt = prompt
        return [{"generated_text": self.answer}]


class RagPromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SemanticEngine.__new__(SemanticEngine)
        self.engine._hf_rag_max_new_tokens = 64
        self.contexts = [
            {
                "filename": "physics.txt",
                "document": "Quantum physics studies particles, energy, and fields.",
            }
        ]

    def test_prompt_requests_a_factual_answer_not_a_citation(self) -> None:
        prompt = self.engine._render_rag_prompt("What does quantum physics study?", self.contexts)

        self.assertIn("not a citation alone", prompt)
        self.assertIn("Question: What does quantum physics study?", prompt)
        self.assertIn("[1] physics.txt", prompt)
        self.assertNotIn("Cite supporting documents", prompt)

    def test_prompt_prefers_context_matching_the_current_query(self) -> None:
        contexts = [
            {
                "filename": "networks.txt",
                "document": (
                    "5G NR supports high-speed mobile broadband. "
                    "6G research explores terahertz frequencies and AI-native networks."
                ),
            }
        ]

        prompt = self.engine._render_rag_prompt("What is 6G?", contexts, max_context_chars=80)

        self.assertIn("6G research explores terahertz frequencies", prompt)
        self.assertNotIn("5G NR supports high-speed mobile broadband", prompt)

    def test_huggingface_answer_is_returned(self) -> None:
        pipeline = _FakePipeline("Quantum physics studies particles and energy.")
        with patch.object(self.engine, "_get_hf_rag_pipeline", return_value=(pipeline, None)):
            answer, error = self.engine._generate_rag_with_huggingface(
                "What does quantum physics study?",
                self.contexts,
            )

        self.assertEqual(answer, "Quantum physics studies particles and energy.")
        self.assertIsNone(error)
        self.assertIn("not a citation alone", pipeline.prompt)

    def test_citation_only_generation_is_reported_as_an_error(self) -> None:
        pipeline = _FakePipeline("[1]")
        with patch.object(self.engine, "_get_hf_rag_pipeline", return_value=(pipeline, None)):
            answer, error = self.engine._generate_rag_with_huggingface(
                "What does quantum physics study?",
                self.contexts,
            )

        self.assertIsNone(answer)
        self.assertEqual(error, "HuggingFace generator returned only citation markers.")

    def test_irrelevant_generated_answer_uses_a_matching_source_excerpt(self) -> None:
        contexts = [
            {
                "filename": "networks.txt",
                "document": (
                    "5G NR supports high-speed mobile broadband. "
                    "6G research explores terahertz frequencies and AI-native networks."
                ),
            }
        ]
        with (
            patch.object(self.engine, "semantic_search", return_value=contexts),
            patch.object(
                self.engine,
                "_generate_rag_answer",
                return_value=("A generic answer unrelated to the question.", "huggingface", None),
            ),
        ):
            response = self.engine.answer_query("What is 6G?")

        self.assertIn("6G research explores", response["answer"])
        self.assertEqual(response["generator"], "extractive")

    def test_email_question_returns_only_matching_email_address(self) -> None:
        contexts = [
            {
                "filename": "other.txt",
                "document": "Unrelated contact is other.person@example.com.",
            },
            {
                "filename": "resume.pdf",
                "document": "ASHEELA RAMU +91 9381816110 ramuashila@gmail.com",
            },
        ]
        with (
            patch.object(self.engine, "semantic_search", return_value=contexts),
            patch.object(self.engine, "_generate_rag_answer") as generate,
        ):
            response = self.engine.answer_query("What is the email of Ramu Asheela?")

        self.assertEqual(response["answer"], "Email address: ramuashila@gmail.com.")
        self.assertEqual(response["generator"], "extractive")
        generate.assert_not_called()

    def test_email_question_reports_when_no_matching_address_exists(self) -> None:
        contexts = [
            {
                "filename": "other.txt",
                "document": "Unrelated contact is other.person@example.com.",
            }
        ]
        with (
            patch.object(self.engine, "semantic_search", return_value=contexts),
            patch.object(self.engine, "_generate_rag_answer") as generate,
        ):
            response = self.engine.answer_query("What is the email of Ramu Asheela?")

        self.assertIn("No email address matching that person", response["answer"])
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
