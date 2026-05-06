import json
import tempfile
import time
import unittest
from pathlib import Path

import feed
from tests.helper import FeedServer, make_rss


def _entry(title, link, date='Mon, 01 Jan 2024 12:00:00 +0000', source='hn'):
    return {
        'title': title,
        'link': link,
        'date': date,
        'description': f'desc-{title}',
        'content': f'desc-{title}',
        'author': '',
        'source': source,
        '_raw_xml': f'<item><title>{title}</title><link>{link}</link></item>',
    }


class CacheTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self._orig_articles = feed.ARTICLES_DIR
        self._orig_state = feed.STATE_FILE
        feed.ARTICLES_DIR = root / 'articles'
        (root / 'config').mkdir()
        feed.STATE_FILE = root / 'config' / 'state.json'

    def tearDown(self):
        feed.ARTICLES_DIR = self._orig_articles
        feed.STATE_FILE = self._orig_state
        self._tmp.cleanup()


class CacheSaveLoadTests(CacheTestBase):
    def test_save_creates_file_with_metadata(self):
        e = _entry('A', 'https://ex.com/a')
        feed.cache_save_entry(e)
        path = feed._article_cache_path('hn', feed.get_article_id('https://ex.com/a'))
        self.assertTrue(path.exists())
        doc = json.loads(path.read_text())
        self.assertEqual(doc['url'], 'https://ex.com/a')
        self.assertEqual(doc['source'], 'hn')
        self.assertIn('first_seen_at', doc)
        self.assertIn('last_seen_at', doc)
        self.assertEqual(doc['first_seen_at'], doc['last_seen_at'])
        self.assertIn('<item>', doc['raw_xml'])
        self.assertEqual(doc['parsed']['title'], 'A')
        self.assertNotIn('_raw_xml', doc['parsed'])  # internal key stripped

    def test_resave_updates_last_seen_at_only(self):
        e = _entry('A', 'https://ex.com/a')
        feed.cache_save_entry(e)
        path = feed._article_cache_path('hn', feed.get_article_id('https://ex.com/a'))
        first = json.loads(path.read_text())

        time.sleep(0.01)  # ensure timestamp differs
        # Re-fetch could yield a re-parsed entry with slightly different fields;
        # verify first_seen_at and raw_xml are preserved.
        e2 = _entry('A (edited)', 'https://ex.com/a')
        e2['_raw_xml'] = '<item>different</item>'
        feed.cache_save_entry(e2)
        second = json.loads(path.read_text())

        self.assertEqual(second['first_seen_at'], first['first_seen_at'])
        self.assertNotEqual(second['last_seen_at'], first['last_seen_at'])
        self.assertEqual(second['raw_xml'], first['raw_xml'])
        self.assertEqual(second['parsed']['title'], 'A')  # parsed preserved too

    def test_load_returns_entries_across_sources(self):
        feed.cache_save_entry(_entry('A1', 'https://ex.com/a1', source='hn'))
        feed.cache_save_entry(_entry('A2', 'https://ex.com/a2', source='hn'))
        feed.cache_save_entry(_entry('B1', 'https://ex.com/b1', source='lwn'))
        loaded = feed.cache_load_entries()
        titles = sorted(e['title'] for e in loaded)
        self.assertEqual(titles, ['A1', 'A2', 'B1'])
        sources = {e['title']: e['source'] for e in loaded}
        self.assertEqual(sources, {'A1': 'hn', 'A2': 'hn', 'B1': 'lwn'})

    def test_load_returns_empty_when_dir_missing(self):
        # ARTICLES_DIR not yet created
        self.assertEqual(feed.cache_load_entries(), [])

    def test_safe_dirname_handles_special_chars(self):
        e = _entry('X', 'https://ex.com/x', source='news.example.com/path')
        feed.cache_save_entry(e)
        # Source dir name should not contain '/'
        loaded = feed.cache_load_entries()
        self.assertEqual(len(loaded), 1)

    def test_save_entries_returns_true_on_success(self):
        ok = feed.cache_save_entries([_entry('A', 'https://ex.com/a')])
        self.assertTrue(ok)

    def test_save_entries_returns_false_when_a_write_fails(self):
        # Place a plain file where the source directory would be created;
        # mkdir(parents=True, exist_ok=True) raises FileExistsError (an OSError).
        feed.ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
        (feed.ARTICLES_DIR / 'blocked').write_text('blocking')
        e = _entry('A', 'https://ex.com/a', source='blocked')
        ok = feed.cache_save_entries([e])
        self.assertFalse(ok)

    def test_save_entries_skips_entries_without_link(self):
        # Missing-link entries are skipped, not failed
        ok = feed.cache_save_entries([{'title': 'no link'}])
        self.assertTrue(ok)

    def test_load_raises_feed_error_on_corrupt_file(self):
        # Plant a corrupt JSON file in the cache
        feed.ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
        src = feed.ARTICLES_DIR / 'hn'
        src.mkdir()
        (src / 'broken.json').write_text('{not valid json')
        with self.assertRaises(feed.FeedError) as ctx:
            feed.cache_load_entries()
        self.assertIn('corrupt cache file', str(ctx.exception))

    def test_save_raises_feed_error_when_existing_file_is_corrupt(self):
        # First save creates a file; corrupt it; resave must raise FeedError
        e = _entry('A', 'https://ex.com/a')
        feed.cache_save_entry(e)
        path = feed._article_cache_path('hn', feed.get_article_id('https://ex.com/a'))
        path.write_text('{not valid json')
        with self.assertRaises(feed.FeedError):
            feed.cache_save_entry(e)


class CacheDisappearanceTests(CacheTestBase):
    def test_dropped_article_persists_in_cache(self):
        feed.cache_save_entry(_entry('A', 'https://ex.com/a'))
        feed.cache_save_entry(_entry('B', 'https://ex.com/b'))
        feed.cache_save_entry(_entry('C', 'https://ex.com/c'))
        # Second fetch only contains A and B; C disappeared from live feed
        feed.cache_save_entry(_entry('A', 'https://ex.com/a'))
        feed.cache_save_entry(_entry('B', 'https://ex.com/b'))

        loaded = feed.cache_load_entries()
        titles = sorted(e['title'] for e in loaded)
        self.assertEqual(titles, ['A', 'B', 'C'])

    def test_new_article_added_after_disappearance(self):
        for t, u in [('A', 'a'), ('B', 'b'), ('C', 'c')]:
            feed.cache_save_entry(_entry(t, f'https://ex.com/{u}'))
        # Next fetch: A dropped, D appeared
        for t, u in [('B', 'b'), ('C', 'c'), ('D', 'd')]:
            feed.cache_save_entry(_entry(t, f'https://ex.com/{u}'))

        loaded = feed.cache_load_entries()
        titles = sorted(e['title'] for e in loaded)
        self.assertEqual(titles, ['A', 'B', 'C', 'D'])

    def test_read_flag_persists_after_disappearance(self):
        # Article ends up in cache, marked read in state. Then it disappears
        # from the live feed. Reading the cached form should still report read.
        e = _entry('A', 'https://ex.com/a')
        feed.cache_save_entry(e)

        state = {'articles': {'sources': {}}}
        feed.toggle_article_read(state, e)
        self.assertTrue(feed.is_article_read(state, e))

        # Simulate re-fetches that no longer contain A — cache is unchanged
        feed.cache_save_entry(_entry('B', 'https://ex.com/b'))
        feed.cache_save_entry(_entry('C', 'https://ex.com/c'))

        # Reload from cache and check the cached A is still flagged read
        loaded = feed.cache_load_entries()
        a_cached = next(e for e in loaded if e['title'] == 'A')
        self.assertTrue(feed.is_article_read(state, a_cached))


class CacheWithFetchTests(CacheTestBase):
    def test_failed_feed_doesnt_lose_cached_articles(self):
        # Pre-populate cache with an article from a feed we won't reach this run
        feed.cache_save_entry(_entry('Old', 'https://ex.com/old', source='down'))

        body = make_rss(
            [
                {
                    'title': 'Fresh',
                    'link': 'https://ex.com/fresh',
                    'pubDate': 'Mon, 01 Jan 2024 12:00:00 +0000',
                    'description': 'd',
                }
            ]
        )
        with FeedServer(body) as good:
            feeds = {
                'up': good.url,
                'down': 'http://127.0.0.1:1/nope',  # closed port → fast failure
            }
            fetched, ok = feed.update_cache_from_feeds(feeds)

        # Only the 'up' feed produced live entries
        self.assertEqual([e['title'] for e in fetched], ['Fresh'])
        # All cache writes for the entries that did come through succeeded
        self.assertTrue(ok)
        # But the cache still has the old article from the down feed
        loaded_titles = sorted(e['title'] for e in feed.cache_load_entries())
        self.assertEqual(loaded_titles, ['Fresh', 'Old'])

    def test_update_mode_writes_cache(self):
        body = make_rss(
            [
                {
                    'title': 'one',
                    'link': 'https://ex.com/1',
                    'pubDate': 'Mon, 01 Jan 2024 12:00:00 +0000',
                    'description': 'd',
                },
                {
                    'title': 'two',
                    'link': 'https://ex.com/2',
                    'pubDate': 'Tue, 02 Jan 2024 12:00:00 +0000',
                    'description': 'd',
                },
            ]
        )
        with FeedServer(body) as server:
            feed.update_cache_from_feeds({'src': server.url})
        loaded = feed.cache_load_entries()
        self.assertEqual(sorted(e['title'] for e in loaded), ['one', 'two'])

    def test_resave_via_fetch_preserves_first_seen_at(self):
        body = make_rss(
            [
                {
                    'title': 'A',
                    'link': 'https://ex.com/a',
                    'pubDate': 'Mon, 01 Jan 2024 12:00:00 +0000',
                    'description': 'd',
                }
            ]
        )
        with FeedServer(body) as server:
            feed.update_cache_from_feeds({'s': server.url})
            path = feed._article_cache_path('s', feed.get_article_id('https://ex.com/a'))
            first_seen = json.loads(path.read_text())['first_seen_at']
            time.sleep(0.01)
            feed.update_cache_from_feeds({'s': server.url})
            doc2 = json.loads(path.read_text())
        self.assertEqual(doc2['first_seen_at'], first_seen)
        self.assertNotEqual(doc2['last_seen_at'], first_seen)


class StateMigrationTests(unittest.TestCase):
    def test_legacy_ids_with_slash_are_migrated(self):
        state = {
            'articles': {
                'sources': {
                    'ex.com': {'aGVsbG8/d29ybGQ=': {'markedRead': True}},
                    'other.com': {'YWJjK2Rlaw==': {'markedRead': False}},
                }
            }
        }
        migrated = feed._migrate_state_ids(state)
        ex = migrated['articles']['sources']['ex.com']
        other = migrated['articles']['sources']['other.com']
        self.assertIn('aGVsbG8_d29ybGQ=', ex)
        self.assertIn('YWJjK2Rlaw==', other)  # no '/' or '+' so unchanged
        # No legacy keys remain
        for src in migrated['articles']['sources'].values():
            for k in src:
                self.assertNotIn('/', k)
                self.assertNotIn('+', k)


if __name__ == '__main__':
    unittest.main()
