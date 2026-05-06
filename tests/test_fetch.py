import unittest

import feed
from tests.helper import FeedServer, make_atom, make_rss


class FetchFeedTests(unittest.TestCase):
    def test_fetches_and_parses_rss(self):
        body = make_rss(
            [
                {
                    "title": "A",
                    "link": "https://ex.com/a",
                    "pubDate": "Mon, 01 Jan 2024 12:00:00 +0000",
                    "description": "first",
                },
                {
                    "title": "B",
                    "link": "https://ex.com/b",
                    "pubDate": "Tue, 02 Jan 2024 12:00:00 +0000",
                    "description": "second",
                },
            ]
        )
        with FeedServer(body) as server:
            tree = feed.fetch_feed(server.url)
            entries = feed.get_entries(tree)
        self.assertEqual([e["title"] for e in entries], ["A", "B"])

    def test_fetches_and_parses_atom(self):
        body = make_atom(
            [
                {
                    "title": "Atom A",
                    "link": "https://ex.com/atom/a",
                    "updated": "2024-02-03T10:00:00Z",
                    "content": "body A",
                },
            ]
        )
        with FeedServer(body, content_type="application/atom+xml; charset=utf-8") as server:
            tree = feed.fetch_feed(server.url)
            entries = feed.get_entries(tree)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["title"], "Atom A")

    def test_cookies_are_sent_in_header(self):
        body = make_rss([{"title": "t", "link": "https://ex.com/x", "description": "d"}])
        with FeedServer(body) as server:
            feed.fetch_feed(server.url, cookies="session=abc; user=42")
            self.assertEqual(len(server.received_headers), 1)
            self.assertEqual(server.received_headers[0].get("Cookie"), "session=abc; user=42")

    def test_no_cookie_header_when_not_provided(self):
        body = make_rss([{"title": "t", "link": "https://ex.com/x", "description": "d"}])
        with FeedServer(body) as server:
            feed.fetch_feed(server.url)
            self.assertNotIn("Cookie", server.received_headers[0])


class FetchAllFeedsTests(unittest.TestCase):
    def test_merges_and_sorts_by_date_descending(self):
        body_old = make_rss(
            [
                {
                    "title": "old",
                    "link": "https://ex.com/old",
                    "pubDate": "Mon, 01 Jan 2024 12:00:00 +0000",
                    "description": "d",
                },
            ]
        )
        body_new = make_rss(
            [
                {
                    "title": "new",
                    "link": "https://ex.com/new",
                    "pubDate": "Wed, 01 Jan 2025 12:00:00 +0000",
                    "description": "d",
                },
            ]
        )
        with FeedServer(body_old) as old, FeedServer(body_new) as new:
            entries = feed.fetch_all_feeds({"old": old.url, "new": new.url})
        titles = [e["title"] for e in entries]
        self.assertEqual(titles, ["new", "old"])
        # Source tag attached
        sources = {e["title"]: e["source"] for e in entries}
        self.assertEqual(sources, {"new": "new", "old": "old"})

    def test_dict_feed_config_with_cookies(self):
        body = make_rss(
            [
                {
                    "title": "t",
                    "link": "https://ex.com/x",
                    "description": "d",
                    "pubDate": "Mon, 01 Jan 2024 12:00:00 +0000",
                }
            ]
        )
        with FeedServer(body) as server:
            feeds = {"s": {"url": server.url, "cookies": "k=v"}}
            entries = feed.fetch_all_feeds(feeds)
            self.assertEqual(len(entries), 1)
            self.assertEqual(server.received_headers[0].get("Cookie"), "k=v")


if __name__ == "__main__":
    unittest.main()
