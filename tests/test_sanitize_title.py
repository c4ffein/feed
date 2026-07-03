import unittest
import xml.etree.ElementTree as ET

import feed


class SanitizeTitleTests(unittest.TestCase):
    def test_newline_becomes_space(self):
        self.assertEqual(feed.sanitize_title("hello\nworld"), "hello world")

    def test_carriage_return_and_tab_become_space(self):
        self.assertEqual(feed.sanitize_title("a\rb\tc"), "a b c")

    def test_whitespace_runs_collapse_and_strip(self):
        self.assertEqual(feed.sanitize_title("  a\n\n\tb  "), "a b")

    def test_printable_ascii_preserved(self):
        s = "Hello, World! (2024) #1 @ 50% -/+ [x]"
        self.assertEqual(feed.sanitize_title(s), s)

    def test_latin1_accented_letters_preserved(self):
        s = "café résumé naïve àÿ"
        self.assertEqual(feed.sanitize_title(s), s)

    def test_cjk_becomes_replacement(self):
        self.assertEqual(feed.sanitize_title("日本"), "��")

    def test_emoji_becomes_replacement(self):
        self.assertEqual(feed.sanitize_title("party🎉time"), "party�time")

    def test_symbols_outside_allowlist_become_replacement(self):
        # euro (U+20AC), copyright (0xA9), and the deliberately-excluded
        # multiplication sign (0xD7) are all outside the allowlist
        self.assertEqual(feed.sanitize_title("10€ © ×"), "10� � �")  # noqa: RUF001 - 0xD7 boundary is the point

    def test_control_char_becomes_replacement(self):
        self.assertEqual(feed.sanitize_title("a\x07b"), "a�b")

    def test_del_and_c1_controls_become_replacement(self):
        self.assertEqual(feed.sanitize_title("x\x7fy\x9fz"), "x�y�z")

    def test_lone_surrogate_becomes_replacement(self):
        self.assertEqual(feed.sanitize_title("a" + chr(0xD800) + "b"), "a�b")

    def test_empty_stays_empty(self):
        self.assertEqual(feed.sanitize_title(""), "")

    def test_applied_in_rss_parsing(self):
        xml = "<rss><channel><item><title>multi\nline\ttitle</title><link>http://e/x</link></item></channel></rss>"
        entries = feed.get_rss_entries(ET.ElementTree(ET.fromstring(xml)))
        self.assertEqual(entries[0]["title"], "multi line title")
        self.assertNotIn("\n", entries[0]["title"])


if __name__ == "__main__":
    unittest.main()
