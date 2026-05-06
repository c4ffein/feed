import io
import unittest
from xml.etree import ElementTree as ET

import feed
from tests.helper import make_atom, make_rss


def _parse(xml_str):
    return ET.parse(io.StringIO(xml_str))


class RSSParserTests(unittest.TestCase):
    def test_extracts_basic_fields(self):
        body = make_rss(
            [
                {
                    'title': 'Hello',
                    'link': 'https://ex.com/a',
                    'pubDate': 'Mon, 01 Jan 2024 12:00:00 +0000',
                    'description': '<p>desc</p>',
                    'author': 'alice@ex.com',
                },
            ]
        )
        entries = feed.get_entries(_parse(body))
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e['title'], 'Hello')
        self.assertEqual(e['link'], 'https://ex.com/a')
        self.assertEqual(e['author'], 'alice@ex.com')
        self.assertIn('desc', e['description'])

    def test_dc_creator_preferred_over_author(self):
        body = make_rss(
            [
                {
                    'title': 't',
                    'link': 'https://ex.com/a',
                    'description': 'd',
                    'author': 'fallback',
                    'dc_creator': 'Real Name',
                },
            ]
        )
        entries = feed.get_entries(_parse(body))
        self.assertEqual(entries[0]['author'], 'Real Name')

    def test_content_encoded_used_when_present(self):
        body = make_rss(
            [
                {
                    'title': 't',
                    'link': 'https://ex.com/a',
                    'description': 'short',
                    'content_encoded': '<p>full content</p>',
                },
            ]
        )
        entries = feed.get_entries(_parse(body))
        self.assertIn('full content', entries[0]['content'])

    def test_falls_back_to_description_when_no_content(self):
        body = make_rss(
            [
                {'title': 't', 'link': 'https://ex.com/a', 'description': 'just desc'},
            ]
        )
        entries = feed.get_entries(_parse(body))
        self.assertEqual(entries[0]['content'], entries[0]['description'])


class AtomParserTests(unittest.TestCase):
    def test_extracts_basic_fields(self):
        body = make_atom(
            [
                {
                    'title': 'Atom post',
                    'link': 'https://ex.com/atom/1',
                    'updated': '2024-02-03T10:00:00Z',
                    'content': '<p>body</p>',
                    'author_name': 'Bob',
                },
            ]
        )
        entries = feed.get_entries(_parse(body))
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual(e['title'], 'Atom post')
        self.assertEqual(e['link'], 'https://ex.com/atom/1')
        self.assertEqual(e['author'], 'Bob')
        self.assertIn('body', e['content'])

    def test_published_preferred_over_updated(self):
        body = make_atom(
            [
                {
                    'title': 't',
                    'link': 'https://ex.com/a',
                    'published': '2024-03-01T00:00:00Z',
                    'updated': '2024-04-01T00:00:00Z',
                    'summary': 's',
                },
            ]
        )
        entries = feed.get_entries(_parse(body))
        self.assertEqual(entries[0]['date'], '2024-03-01T00:00:00Z')

    def test_summary_used_when_no_content(self):
        body = make_atom(
            [
                {'title': 't', 'link': 'https://ex.com/a', 'summary': 'sum only'},
            ]
        )
        entries = feed.get_entries(_parse(body))
        self.assertEqual(entries[0]['content'], 'sum only')


if __name__ == '__main__':
    unittest.main()
