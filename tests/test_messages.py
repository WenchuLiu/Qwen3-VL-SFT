import unittest

from qwen3vl_sft.data.messages import build_messages


class MessagesTest(unittest.TestCase):
    def test_media_placeholders_are_checked(self):
        with self.assertRaisesRegex(ValueError, "more <image>"):
            build_messages(
                {
                    "image": ["a.jpg"],
                    "conversations": [{"from": "human", "value": "<image>\n<image>"}],
                },
            )
        with self.assertRaisesRegex(ValueError, "not referenced"):
            build_messages(
                {
                    "image": ["a.jpg", "b.jpg"],
                    "conversations": [{"from": "human", "value": "<image>"}],
                },
            )

    def test_roles_are_normalized_to_hf_roles(self):
        messages = build_messages(
            {
                "image": "a.jpg",
                "conversations": [
                    {"from": "system", "value": "system"},
                    {"from": "human", "value": "<image> question"},
                    {"from": "gpt", "value": "answer"},
                ],
            },
        )
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
        self.assertEqual(messages[1]["content"][0]["type"], "image")


if __name__ == "__main__":
    unittest.main()
