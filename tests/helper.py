"""Test helpers: a local HTTP server that serves configurable RSS/Atom feeds."""

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from xml.sax.saxutils import escape


class FeedServer:
    """Context manager that serves a fixed XML body on a local random port.

    Captures incoming request headers in `received_headers` so tests can
    assert on cookies, user-agent, etc.
    """

    def __init__(self, body, content_type="application/rss+xml; charset=utf-8"):
        self.body = body.encode() if isinstance(body, str) else body
        self.content_type = content_type
        self.received_headers = []
        self._server = None
        self._thread = None
        self.url = None

    def __enter__(self):
        body = self.body
        ct = self.content_type
        received = self.received_headers

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                received.append(dict(self.headers))
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args, **kwargs):
                pass  # silence default access log

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address
        self.url = f"http://{host}:{port}/"
        return self

    def __exit__(self, exc_type, exc, tb):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


def _wrap_cdata(text):
    return f"<![CDATA[{text}]]>"


def make_rss(items, channel_title="Test Feed", channel_link="https://example.com/"):
    """Build an RSS 2.0 XML body from a list of item dicts.

    Recognized item keys: title, link, pubDate, description, author,
    dc_creator, content_encoded.
    """
    item_xml = []
    for it in items:
        parts = []
        if "title" in it:
            parts.append(f"<title>{escape(it['title'])}</title>")
        if "link" in it:
            parts.append(f"<link>{escape(it['link'])}</link>")
        if "pubDate" in it:
            parts.append(f"<pubDate>{escape(it['pubDate'])}</pubDate>")
        if "description" in it:
            parts.append(f"<description>{_wrap_cdata(it['description'])}</description>")
        if "author" in it:
            parts.append(f"<author>{escape(it['author'])}</author>")
        if "dc_creator" in it:
            parts.append(f"<dc:creator>{escape(it['dc_creator'])}</dc:creator>")
        if "content_encoded" in it:
            parts.append(f"<content:encoded>{_wrap_cdata(it['content_encoded'])}</content:encoded>")
        item_xml.append("<item>" + "".join(parts) + "</item>")

    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss version="2.0" '
        'xmlns:content="http://purl.org/rss/1.0/modules/content/" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/">'
        "<channel>"
        f"<title>{escape(channel_title)}</title>"
        f"<link>{escape(channel_link)}</link>"
        f"<description>test</description>"
        f"{''.join(item_xml)}"
        "</channel></rss>"
    )


def make_atom(entries, feed_title="Test Atom Feed"):
    """Build an Atom XML body from a list of entry dicts.

    Recognized entry keys: title, link (href), updated, published,
    summary, content, author_name.
    """
    entry_xml = []
    for e in entries:
        parts = []
        if "title" in e:
            parts.append(f"<title>{escape(e['title'])}</title>")
        if "link" in e:
            parts.append(f'<link rel="alternate" href="{escape(e["link"])}"/>')
        if "updated" in e:
            parts.append(f"<updated>{escape(e['updated'])}</updated>")
        if "published" in e:
            parts.append(f"<published>{escape(e['published'])}</published>")
        if "summary" in e:
            parts.append(f"<summary>{escape(e['summary'])}</summary>")
        if "content" in e:
            parts.append(f'<content type="html">{_wrap_cdata(e["content"])}</content>')
        if "author_name" in e:
            parts.append(f"<author><name>{escape(e['author_name'])}</name></author>")
        entry_xml.append("<entry>" + "".join(parts) + "</entry>")

    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        f"<title>{escape(feed_title)}</title>"
        f"<id>urn:test</id><updated>2024-01-01T00:00:00Z</updated>"
        f"{''.join(entry_xml)}"
        "</feed>"
    )
