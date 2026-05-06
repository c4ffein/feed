import unittest

import feed


class HtmlToTextTests(unittest.TestCase):
    def test_strips_script_and_style(self):
        out = feed.html_to_text("<p>keep</p><script>alert(1)</script><style>x{}</style>")
        self.assertIn("keep", out)
        self.assertNotIn("alert", out)
        self.assertNotIn("x{}", out)

    def test_renders_lists(self):
        out = feed.html_to_text("<ul><li>one</li><li>two</li></ul>")
        self.assertIn("one", out)
        self.assertIn("two", out)
        self.assertIn("•", out)

    def test_unescapes_entities(self):
        # &amp; survives as &; &lt;c&gt; gets unescaped to <c> and then
        # stripped as an unknown HTML tag (intended behavior).
        out = feed.html_to_text("<p>a &amp; b &lt;c&gt; tail</p>")
        self.assertIn("a & b", out)
        self.assertIn("tail", out)
        self.assertNotIn("<c>", out)

    def test_image_alt_rendered(self):
        out = feed.html_to_text('<img alt="diagram" src="x.png">')
        self.assertIn("diagram", out)


class ParseDateTests(unittest.TestCase):
    def test_rfc822(self):
        d = feed.parse_date("Mon, 01 Jan 2024 12:00:00 +0000")
        self.assertIsNotNone(d)
        self.assertEqual(d.year, 2024)

    def test_iso8601_with_z(self):
        d = feed.parse_date("2024-02-03T10:00:00Z")
        self.assertIsNotNone(d)
        self.assertEqual((d.year, d.month, d.day), (2024, 2, 3))

    def test_empty_returns_none(self):
        self.assertIsNone(feed.parse_date(""))

    def test_garbage_returns_none(self):
        self.assertIsNone(feed.parse_date("not a date"))


class DisableColorsTests(unittest.TestCase):
    def test_disable_colors_clears_codes(self):
        # Snapshot to restore after test (module-level globals)
        snapshot = {
            k: getattr(feed, k)
            for k in ("RED", "GREEN", "PURP", "DIM", "WHITE", "BOLD", "ITALIC", "REVERSE", "BG_WHITE", "BLACK", "RESET")
        }
        try:
            feed.disable_colors()
            for k in snapshot:
                self.assertEqual(getattr(feed, k), "", f"{k} should be empty")
        finally:
            for k, v in snapshot.items():
                setattr(feed, k, v)


if __name__ == "__main__":
    unittest.main()
