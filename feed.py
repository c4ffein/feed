#!/usr/bin/env python3
"""feed - KISS cli RSS reader with HTML rendering"""

import argparse
import base64
import json
import select
import shutil
import sys
import termios
import textwrap
import tty
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

# ANSI escape codes
RED = '\033[31m'
GREEN = '\033[32m'
PURP = '\033[34m'
DIM = '\033[90m'
WHITE = '\033[39m'
BOLD = '\033[1m'
ITALIC = '\033[3m'
REVERSE = '\033[7m'
BG_WHITE = '\033[47m'
BLACK = '\033[30m'
RESET = '\033[0m'
CLEAR_SCREEN = '\033[2J\033[H'
HOME = '\033[H'
CLEAR_LINE = '\033[K'
CLEAR_TO_END = '\033[J'


def disable_colors():
    """Wipe all ANSI codes to empty strings (for --no-color)."""
    global RED, GREEN, PURP, DIM, WHITE
    global BOLD, ITALIC, REVERSE, BG_WHITE, BLACK, RESET
    RED = GREEN = PURP = DIM = WHITE = ''
    BOLD = ITALIC = REVERSE = BG_WHITE = BLACK = RESET = ''


class FeedError(Exception):
    pass


class Term:
    """Static utility class for terminal operations."""

    @staticmethod
    def getch() -> str:
        """Read a single character/escape sequence from stdin.

        Uses select with 0.1s timeout after ESC to distinguish
        arrow keys (ESC + sequence) from standalone Escape key.
        """
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            # If ESC, check for arrow key sequence (else falls through to standalone ESC)
            if ch == '\x1b' and select.select([sys.stdin], [], [], 0.1)[0]:
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A':
                        return 'UP'
                    if ch3 == 'B':
                        return 'DOWN'
                    if ch3 == 'C':
                        return 'RIGHT'
                    if ch3 == 'D':
                        return 'LEFT'
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    @staticmethod
    def size() -> tuple[int, int]:
        """Return terminal (columns, lines)."""
        sz = shutil.get_terminal_size()
        return sz.columns, sz.lines

    @staticmethod
    def write(text: str) -> None:
        """Write text directly to stdout."""
        sys.stdout.write(text)
        sys.stdout.flush()


class Screen:
    """Buffered screen rendering to reduce SSH flicker."""

    def __init__(self):
        self.buf: list[str] = []

    def clear(self) -> None:
        """Add clear screen sequence to buffer."""
        self.buf.append(CLEAR_SCREEN)

    def home(self) -> None:
        """Add cursor home sequence to buffer."""
        self.buf.append(HOME)

    def move(self, row: int, col: int) -> None:
        """Add cursor move sequence to buffer."""
        self.buf.append(f'\033[{row};{col}H')

    def write(self, text: str) -> None:
        """Add text to buffer."""
        self.buf.append(text)

    def clear_line(self) -> None:
        """Add clear line sequence to buffer."""
        self.buf.append(CLEAR_LINE)

    def clear_to_end(self) -> None:
        """Add clear to end of screen sequence to buffer."""
        self.buf.append(CLEAR_TO_END)

    def writeln(self, text: str) -> None:
        """Add text with newline and clear line to buffer."""
        self.buf.append(text)
        self.buf.append(CLEAR_LINE)
        self.buf.append('\n')

    def flush(self) -> None:
        """Write entire buffer to stdout in single call, then clear buffer."""
        sys.stdout.write(''.join(self.buf))
        sys.stdout.flush()
        self.buf.clear()


CONFIG_DIR = Path.home() / '.config' / 'feed'
CONFIG_FILE = CONFIG_DIR / 'config.json'
STATE_FILE = CONFIG_DIR / 'state.json'
CACHE_DIR = Path.home() / '.cache' / 'feed'
ARTICLES_DIR = CACHE_DIR / 'articles'

EXAMPLE_FEEDS = {
    'example': 'https://example.com/feed.xml',
    'another_example': 'https://blog.example.org/rss',
    'another_example_with_cookies': {'url': 'https://forum.example.org/feed.xml', 'cookies': 'session_id=abc123'},
}


def load_feeds():
    """Load feeds from config file, or exit with example config."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config = json.load(f)
            return config.get('feeds', {})
    else:
        example = json.dumps({'feeds': EXAMPLE_FEEDS}, indent=2)
        print(f'Config file not found: {CONFIG_FILE}\n')
        print(f'Create it with your feeds:\n\n{example}')
        sys.exit(1)


def get_article_id(url):
    """Generate a filesystem-safe unique ID from URL."""
    return base64.urlsafe_b64encode(url.encode()).decode()


def get_source_from_url(url):
    """Extract domain from URL to use as source key."""
    return urlparse(url).netloc


def _migrate_state_ids(state):
    """Rewrite legacy base64 IDs (with /, +) to urlsafe form."""
    sources = state.get('articles', {}).get('sources', {})
    for src, articles in list(sources.items()):
        if any(('/' in k or '+' in k) for k in articles):
            sources[src] = {k.replace('/', '_').replace('+', '-'): v for k, v in articles.items()}
    return state


def load_state():
    """Load state from state file."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return _migrate_state_ids(json.load(f))
    return {'articles': {'sources': {}}}


def save_state(state):
    """Save state to state file atomically."""
    tmp_file = STATE_FILE.with_suffix('.tmp')
    with open(tmp_file, 'w') as f:
        json.dump(state, f, indent=2)
    tmp_file.rename(STATE_FILE)


def _safe_dirname(name):
    """Replace filesystem-unsafe chars in a source name."""
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in name)


def _article_cache_path(source, article_id):
    return ARTICLES_DIR / _safe_dirname(source) / f'{article_id}.json'


def cache_save_entry(entry):
    """Write or update one article's cache file. Atomic via tmp + rename.

    On first save: writes first_seen_at, last_seen_at, source, url, raw_xml,
    parsed. On re-save: only last_seen_at is updated; raw_xml + parsed are
    preserved as originally captured.
    """
    source = entry['source']
    article_id = get_article_id(entry['link'])
    path = _article_cache_path(source, article_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    raw_xml = entry.get('_raw_xml')
    parsed = {k: v for k, v in entry.items() if not k.startswith('_')}

    if path.exists():
        try:
            with open(path) as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise FeedError(f'corrupt cache file {path}: {e}') from e
        doc['last_seen_at'] = now
    else:
        doc = {
            'first_seen_at': now,
            'last_seen_at': now,
            'source': source,
            'url': entry['link'],
            'raw_xml': raw_xml,
            'parsed': parsed,
        }

    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(doc, f, indent=2)
    tmp.rename(path)


def cache_save_entries(entries):
    """Save many entries; per-entry failures are logged and skipped.

    Returns True iff every (non-skipped) entry was written successfully.
    """
    all_ok = True
    for entry in entries:
        if not entry.get('link'):
            continue
        try:
            cache_save_entry(entry)
        except OSError as e:
            print(f'{RED}cache write failed for {entry.get("link")}: {e}{RESET}')
            all_ok = False
    return all_ok


def cache_load_entries():
    """Walk the article cache and return all parsed entries."""
    entries = []
    if not ARTICLES_DIR.exists():
        return entries
    for src_dir in ARTICLES_DIR.iterdir():
        if not src_dir.is_dir():
            continue
        for art_file in src_dir.iterdir():
            if art_file.suffix != '.json':
                continue
            try:
                with open(art_file) as f:
                    doc = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                raise FeedError(f'corrupt cache file {art_file}: {e}') from e
            parsed = doc.get('parsed') or {}
            parsed.setdefault('source', doc.get('source', src_dir.name))
            entries.append(parsed)
    return entries


def is_article_read(state, entry):
    """Check if an article is marked as read."""
    source = get_source_from_url(entry['link'])
    article_id = get_article_id(entry['link'])
    sources = state.get('articles', {}).get('sources', {})
    return sources.get(source, {}).get(article_id, {}).get('markedRead', False)


def toggle_article_read(state, entry):
    """Toggle an article's read status."""
    source = get_source_from_url(entry['link'])
    article_id = get_article_id(entry['link'])
    if 'articles' not in state:
        state['articles'] = {'sources': {}}
    if 'sources' not in state['articles']:
        state['articles']['sources'] = {}
    if source not in state['articles']['sources']:
        state['articles']['sources'][source] = {}
    current = state['articles']['sources'][source].get(article_id, {}).get('markedRead', False)
    state['articles']['sources'][source][article_id] = {'markedRead': not current}
    save_state(state)


class HTMLToTerminal(HTMLParser):
    """Convert HTML to terminal-formatted text with ANSI codes."""

    def __init__(self, width=80):
        super().__init__()
        self.output = []
        self.width = width
        self.in_pre = False
        self.list_depth = 0
        self.skip_content = False  # for script/style tags

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'iframe'):
            self.skip_content = True
        elif tag in ('strong', 'b'):
            self.output.append(BOLD)
        elif tag in ('em', 'i'):
            self.output.append(ITALIC)
        elif tag == 'a':
            self.output.append(PURP)
        elif tag == 'code':
            self.output.append(GREEN)
        elif tag in ('h1', 'h2', 'h3', 'h4'):
            self.output.append(f'\n\n{BOLD}{PURP}')
        elif tag == 'p':
            self.output.append('\n\n')
        elif tag == 'br':
            self.output.append('\n')
        elif tag in ('ul', 'ol'):
            self.list_depth += 1
            self.output.append('\n')
        elif tag == 'li':
            self.output.append('\n' + '  ' * self.list_depth + '• ')
        elif tag == 'pre':
            self.in_pre = True
            self.output.append(f'\n{DIM}')
        elif tag == 'blockquote':
            self.output.append(f'\n{DIM}  │ ')
        elif tag == 'img':
            alt = dict(attrs).get('alt', 'image')
            self.output.append(f'{DIM}[{alt}]{RESET}')

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'iframe'):
            self.skip_content = False
        elif tag in ('strong', 'b', 'em', 'i', 'a', 'code'):
            self.output.append(RESET)
        elif tag in ('h1', 'h2', 'h3', 'h4'):
            self.output.append(f'{RESET}\n')
        elif tag in ('ul', 'ol'):
            self.list_depth = max(0, self.list_depth - 1)
            self.output.append('\n')
        elif tag == 'pre':
            self.in_pre = False
            self.output.append(RESET)
        elif tag == 'blockquote':
            self.output.append(RESET)

    def handle_data(self, data):
        if self.skip_content:
            return
        if not self.in_pre:
            # Normalize whitespace but preserve leading/trailing space
            has_leading = data and data[0].isspace()
            has_trailing = data and data[-1].isspace()
            data = ' '.join(data.split())
            if data:
                if has_leading:
                    data = ' ' + data
                if has_trailing:
                    data = data + ' '
        if data:
            self.output.append(data)

    def get_text(self):
        text = ''.join(self.output)
        # Clean up excessive newlines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')
        return text.strip()


def html_to_text(html, width=80):
    """Convert HTML string to terminal-formatted text."""
    parser = HTMLToTerminal(width)
    parser.feed(unescape(html))
    return parser.get_text()


def wrap_text(text, width=80):
    """Wrap text preserving paragraph breaks and ANSI codes."""
    paragraphs = text.split('\n\n')
    wrapped = []
    for p in paragraphs:
        lines = p.split('\n')
        wrapped_lines = []
        for line in lines:
            if line.startswith(('  ', '│')):  # preserve indented/quoted
                wrapped_lines.append(line)
            elif len(line) > width:
                wrapped_lines.extend(textwrap.wrap(line, width))
            else:
                wrapped_lines.append(line)
        wrapped.append('\n'.join(wrapped_lines))
    return '\n\n'.join(wrapped)


def fetch_feed(url, cookies=None):
    """Fetch and parse RSS feed."""
    # Safe: ElementTree doesn't process external entities (no XXE),
    # and Python 3.7.1+ limits entity expansion (no billion laughs)
    headers = {'User-Agent': 'Python RSS Reader'}
    if cookies:
        headers['Cookie'] = cookies
    req = Request(url, headers=headers)
    with urlopen(req, timeout=10) as response:
        return ET.parse(response)


def get_rss_entries(tree):
    """Extract entries from RSS feed."""
    entries = []
    ns = {
        'content': 'http://purl.org/rss/1.0/modules/content/',
        'dc': 'http://purl.org/dc/elements/1.1/',
    }

    for item in tree.findall('.//item'):
        # Get author from dc:creator or author element
        author = item.findtext('dc:creator', '', ns) or item.findtext('author', '')

        entry = {
            'title': item.findtext('title', ''),
            'link': item.findtext('link', ''),
            'date': item.findtext('pubDate', ''),
            'description': item.findtext('description', ''),
            'author': author,
        }
        # Try to get full content
        content = item.find('content:encoded', ns)
        if content is not None and content.text:
            entry['content'] = content.text
        else:
            entry['content'] = entry['description']
        entry['_raw_xml'] = ET.tostring(item, encoding='unicode')
        entries.append(entry)
    return entries


def get_atom_entries(root):
    """Extract entries from Atom feed."""
    entries = []
    # Atom namespace
    atom_ns = '{http://www.w3.org/2005/Atom}'

    atom_entries = root.findall(f'.//{atom_ns}entry') or root.findall('.//entry')

    for item in atom_entries:
        # Get link href attribute (Atom uses <link href="..."/>)
        # Prefer rel="alternate", fall back to first link
        link = ''
        for link_elem in item.findall(f'{atom_ns}link') or item.findall('link'):
            if link_elem.get('rel', 'alternate') == 'alternate':
                link = link_elem.get('href', '')
                break
        if not link:
            link_elem = item.find(f'{atom_ns}link')
            if link_elem is None:
                link_elem = item.find('link')
            link = link_elem.get('href', '') if link_elem is not None else ''

        # Get content or summary
        content_elem = item.find(f'{atom_ns}content')
        if content_elem is None:
            content_elem = item.find('content')
        summary_elem = item.find(f'{atom_ns}summary')
        if summary_elem is None:
            summary_elem = item.find('summary')
        content = ''
        if content_elem is not None and content_elem.text:
            content = content_elem.text
        elif summary_elem is not None and summary_elem.text:
            content = summary_elem.text

        # Get author name
        author_elem = item.find(f'{atom_ns}author')
        if author_elem is None:
            author_elem = item.find('author')
        author = ''
        if author_elem is not None:
            author = author_elem.findtext(f'{atom_ns}name', '') or author_elem.findtext('name', '')

        entry = {
            'title': item.findtext(f'{atom_ns}title', '') or item.findtext('title', ''),
            'link': link,
            'date': item.findtext(f'{atom_ns}published', '')
            or item.findtext(f'{atom_ns}updated', '')
            or item.findtext('published', '')
            or item.findtext('updated', ''),
            'description': content,
            'content': content,
            'author': author,
        }
        entry['_raw_xml'] = ET.tostring(item, encoding='unicode')
        entries.append(entry)
    return entries


def get_entries(tree):
    """Extract entries from RSS or Atom feed."""
    root = tree.getroot()
    # Detect Atom feed by root tag
    if root.tag in ('{http://www.w3.org/2005/Atom}feed', 'feed'):
        return get_atom_entries(root)
    return get_rss_entries(tree)


def truncate_title(title, max_width):
    """Truncate title with ellipsis if too long."""
    if len(title) <= max_width:
        return title
    return title[: max_width - 1] + '…'


def list_entries(scr, entries, selected=None, offset=0, height=20, selection_active=True, state=None, width=None):
    """Display list of entries with optional selection highlight."""
    term_width = width if width is not None else Term.size()[0]
    help_hint = '[i/k: move, I/K: ×10, 0-9: jump, Enter: open, r: toggle read, q: quit]'  # noqa: RUF001
    scr.writeln(f'{BOLD}Found {len(entries)} articles:{RESET}  {DIM}{help_hint}{RESET}')
    for i in range(offset, min(offset + height, len(entries))):
        entry = entries[i]
        date = entry['date'].split(' +')[0] if entry['date'] else ''
        source = entry.get('source', '')
        source_str = f'[{source}] ' if source else ''
        # Calculate prefix length: "  8. [source] " = 5 chars + source_str
        prefix_len = 5 + len(source_str)
        max_title_width = term_width - prefix_len - 1
        title = truncate_title(entry['title'], max_title_width)
        is_read = state and is_article_read(state, entry)
        if i == selected:
            if selection_active:
                # White/bright selection
                scr.writeln(f'{BG_WHITE}{BLACK}{i + 1:3}. {source_str}{title}{RESET}')
                scr.writeln(f'{BG_WHITE}{BLACK}     {date}{RESET}')
            else:
                # Dimmed selection (gray)
                scr.writeln(f'{DIM}{REVERSE}{i + 1:3}. {source_str}{title}{RESET}')
                scr.writeln(f'{DIM}{REVERSE}     {date}{RESET}')
        elif is_read:
            # Read articles: title in red
            scr.writeln(f'{PURP}{i + 1:3}.{RESET} {DIM}{source_str}{RESET}{RED}{title}{RESET}')
            scr.writeln(f'     {DIM}{date}{RESET}')
        else:
            scr.writeln(f'{PURP}{i + 1:3}.{RESET} {DIM}{source_str}{RESET}{BOLD}{title}{RESET}')
            scr.writeln(f'     {DIM}{date}{RESET}')


def show_article(entry, width=80):
    """Display a single article."""
    print(f'\n{"─" * width}')
    print(f'{BOLD}{PURP}{entry["title"]}{RESET}')
    author = entry.get('author', '')
    if author:
        print(f'{DIM}by {author}{RESET}')
    print(f'{DIM}{entry["date"]}{RESET}')
    print(f'{PURP}{entry["link"]}{RESET}')
    print(f'{"─" * width}\n')

    content = html_to_text(entry['content'], width)
    print(wrap_text(content, width))
    print(f'\n{"─" * width}')


def draw_number_bar(scr, written_number, active, width):
    """Draw the number input bar at the bottom."""
    label = '> '
    num_str = written_number or ''
    if active:
        scr.writeln(f'{BG_WHITE}{BLACK}{label}{num_str}_{" " * (width - len(label) - len(num_str) - 1)}{RESET}')
    else:
        scr.writeln(f'{DIM}{label}{num_str}{RESET}')
    scr.clear_to_end()  # Clear any leftover lines below


def interactive_mode(entries, width=80):
    """Interactive article browser with vim-style navigation."""
    selected = 0
    _, term_height = Term.size()
    term_height -= 6  # Leave room for header/footer/number bar
    visible_count = max(1, term_height // 2)  # 2 lines per entry
    offset = 0
    written_number = ''
    currently_writing_number = False
    state = load_state()
    scr = Screen()

    scr.clear()
    scr.flush()
    while True:
        # Adjust offset to keep selection visible
        if selected < offset:
            offset = selected
        elif selected >= offset + visible_count:
            offset = selected - visible_count + 1

        scr.home()
        list_entries(
            scr,
            entries,
            selected,
            offset,
            visible_count,
            selection_active=not currently_writing_number,
            state=state,
            width=width,
        )
        draw_number_bar(scr, written_number, currently_writing_number, width)
        scr.flush()

        try:
            key = Term.getch()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if key in ('q', '\x03', '\x1b'):  # q, Ctrl+C, or Escape
            break
        if key in '0123456789':
            written_number += key
            currently_writing_number = True
        elif key in ('\x7f', '\x08'):  # Backspace
            if written_number:
                written_number = written_number[:-1]
            currently_writing_number = True  # Backspace always focuses the number bar
        elif key in ('k', '\x0b', 'DOWN'):  # down (k, Ctrl+K, or arrow)
            selected = min(selected + 1, len(entries) - 1)
            currently_writing_number = False
        elif key in ('i', '\x1e', 'UP'):  # up (i, Ctrl+^, or arrow)
            selected = max(selected - 1, 0)
            currently_writing_number = False
        elif key == 'K':  # down 10
            selected = min(selected + 10, len(entries) - 1)
            currently_writing_number = False
        elif key == 'I':  # up 10
            selected = max(selected - 10, 0)
            currently_writing_number = False
        elif key == 'r':  # Toggle read status
            toggle_article_read(state, entries[selected])
        elif key in ('\r', '\n'):  # Enter
            if currently_writing_number and written_number:
                target = int(written_number) - 1  # Convert to 0-indexed
                if 0 <= target < len(entries):
                    selected = target
                    print(CLEAR_SCREEN, end='')
                    show_article(entries[selected], width)
                    print(f'\n{DIM}Press any key to continue...{RESET}')
                    Term.getch()
                written_number = ''
                currently_writing_number = False
            else:
                print(CLEAR_SCREEN, end='')
                show_article(entries[selected], width)
                print(f'\n{DIM}Press any key to continue...{RESET}')
                Term.getch()


def parse_date(date_str):
    """Parse RSS (RFC 822) or Atom (ISO 8601) date string for sorting."""
    if not date_str:
        return None
    # Try RFC 822 (RSS)
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        pass
    # Try ISO 8601 (Atom)
    try:
        # Handle Z suffix and timezone
        date_str = date_str.replace('Z', '+00:00')
        return datetime.fromisoformat(date_str)
    except Exception:
        return None


def get_feed_config(feed_value):
    """Extract url and cookies from feed config (string or dict)."""
    if isinstance(feed_value, str):
        return feed_value, None
    return feed_value.get('url', ''), feed_value.get('cookies')


def fetch_single_feed(name, feed_value):
    """Fetch a single feed and return (name, entries) or (name, error)."""
    url, cookies = get_feed_config(feed_value)
    try:
        tree = fetch_feed(url, cookies)
        entries = get_entries(tree)
        for entry in entries:
            entry['source'] = name
        return (name, entries, None)
    except Exception as e:
        return (name, [], e)


def fetch_all_feeds(feeds):
    """Fetch all configured feeds concurrently and merge entries sorted by date."""
    all_entries = []
    errors = []

    print(f'{DIM}Fetching {len(feeds)} feeds...{RESET}')

    with ThreadPoolExecutor(max_workers=len(feeds)) as executor:
        futures = {executor.submit(fetch_single_feed, name, feed_value): name for name, feed_value in feeds.items()}

        for future in as_completed(futures):
            name, entries, error = future.result()
            if error:
                errors.append(f'{name}: {error}')
                print(f'{RED}Error fetching {name}: {error}{RESET}')
            else:
                print(f'{DIM}Fetched {name} ({len(entries)} articles){RESET}')
                all_entries.extend(entries)

    if errors and sys.stdin.isatty():
        print(f'\n{DIM}Press any key to continue...{RESET}')
        Term.getch()

    all_entries.sort(key=lambda e: parse_date(e['date']) or datetime.min, reverse=True)
    return all_entries


def _sort_by_date_desc(entries):
    entries.sort(key=lambda e: parse_date(e.get('date', '')) or datetime.min, reverse=True)
    return entries


def update_cache_from_feeds(feeds):
    """Fetch all configured feeds and persist new articles to the cache.

    Returns (fetched_entries, all_writes_ok).
    """
    fetched = fetch_all_feeds(feeds)
    ok = cache_save_entries(fetched)
    return fetched, ok


def main():
    feeds = load_feeds()
    parser = argparse.ArgumentParser(description='feed - KISS cli RSS reader with HTML rendering')
    parser.add_argument(
        'feed', nargs='?', default=None, help=f'Feed URL or shortcut: {", ".join(feeds.keys())} (default: all)'
    )
    parser.add_argument('-l', '--list', action='store_true', help='List articles without interactive mode')
    parser.add_argument('-n', '--number', type=int, help='Show article N directly')
    parser.add_argument('-w', '--width', type=int, default=80, help='Terminal width (default: 80)')
    parser.add_argument('--no-color', action='store_true', help='Disable ANSI colors')
    parser.add_argument('--update', action='store_true', help='Fetch all feeds into the cache and exit (for cron).')
    args = parser.parse_args()

    if args.no_color:
        disable_colors()

    if args.update:
        _, ok = update_cache_from_feeds(feeds)
        sys.exit(0 if ok else 1)

    if args.feed is None:
        update_cache_from_feeds(feeds)
        entries = _sort_by_date_desc(cache_load_entries())
    elif args.feed in feeds:
        update_cache_from_feeds({args.feed: feeds[args.feed]})
        cached = [e for e in cache_load_entries() if e.get('source') == args.feed]
        entries = _sort_by_date_desc(cached)
    else:
        # Ad-hoc URL: live fetch, no cache
        if not args.feed.startswith('http'):
            print(f'Unknown feed: {args.feed}')
            print(f'Available shortcuts: {", ".join(feeds.keys())}')
            sys.exit(1)
        print(f'{DIM}Fetching {args.feed}...{RESET}')
        try:
            tree = fetch_feed(args.feed)
            entries = _sort_by_date_desc(get_entries(tree))
        except Exception as e:
            print(f'Error fetching feed: {e}')
            sys.exit(1)

    if not entries:
        print('No entries found.')
        sys.exit(1)

    if args.number:
        if 1 <= args.number <= len(entries):
            show_article(entries[args.number - 1], args.width)
        else:
            print(f'Invalid article number. Choose 1-{len(entries)}')
            sys.exit(1)
    elif args.list:
        scr = Screen()
        list_entries(scr, entries, height=len(entries), width=args.width)
        scr.flush()
    else:
        interactive_mode(entries, args.width)


if __name__ == '__main__':
    try:
        main()
    except FeedError as e:
        print(f'{RED}\n  !!  {e}  !!  \n{RESET}')
        sys.exit(1)
    except Exception:
        raise
