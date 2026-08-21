import unittest

from src.presentation import (
    escape_markdown,
    format_index_status,
    format_processing_status,
)


class AppFormattingTests(unittest.TestCase):
    def test_markdown_control_characters_in_filename_are_escaped(self):
        escaped = escape_markdown("[unsafe](link).pdf")

        self.assertEqual(escaped, r"\[unsafe\]\(link\)\.pdf")

    def test_failed_batch_explains_that_existing_index_was_kept(self):
        status = format_processing_status(
            [
                {
                    "success": False,
                    "file": "empty.txt",
                    "error": "file is empty",
                }
            ],
            total_chunks=0,
        )

        self.assertIn("Nothing was indexed", status)
        self.assertIn("kept unchanged", status)

    def test_success_status_reports_documents_and_chunks(self):
        status = format_processing_status(
            [
                {
                    "success": True,
                    "file": "guide.pdf",
                    "chunks": 12,
                }
            ],
            total_chunks=12,
        )

        self.assertIn("Knowledge base ready", status)
        self.assertIn("12 searchable chunks", status)

    def test_existing_index_status_reports_persisted_chunks(self):
        status = format_index_status(27)

        self.assertIn("Existing knowledge base ready", status)
        self.assertIn("27 searchable chunks", status)


if __name__ == "__main__":
    unittest.main()
