#!/usr/bin/env python3
"""Zero-dependency CLI RSS reader with HTML rendering."""

import base64
import json
import shutil
import sys
import termios
import textwrap
import tty
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, Request
from xml.etree import ElementTree as ET

# ANSI escape codes
from enum import Enum

colors = {"RED": "31", "GREEN": "32", "PURP": "34", "DIM": "90", "WHITE": "39"}
Color = Enum("Color", [(k, f"\033[{v}m") for k, v in colors.items()])

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


def getch():
    """Read a single character from stdin without waiting for Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

CONFIG_DIR = Path.home() / '.config' / 'feed'
CONFIG_FILE = CONFIG_DIR / 'config.json'
STATE_FILE = CONFIG_DIR / 'state.json'

EXAMPLE_FEEDS = {
    'example': 'https://example.com/feed.xml',
    'another_example': 'https://blog.example.org/rss',
    'another_example_with_cookies': {
        'url': 'https://forum.example.org/feed.xml',
        'cookies': 'session_id=abc123'
    },
}


def load_feeds():
    """Load feeds from config file, or exit with example config."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config = json.load(f)
            return config.get('feeds', {})
    else:
        example = json.dumps({'feeds': EXAMPLE_FEEDS}, indent=2)
        print(f"Config file not found: {CONFIG_FILE}\n")
        print(f"Create it with your feeds:\n\n{example}")
        sys.exit(1)


FEEDS = load_feeds()


def get_article_id(url):
    """Generate a unique ID from URL using base64 encoding."""
    return base64.b64encode(url.encode()).decode()


def get_source_from_url(url):
    """Extract domain from URL to use as source key."""
    return urlparse(url).netloc


def load_state():
    """Load state from state file."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"articles": {"sources": {}}}


def save_state(state):
    """Save state to state file atomically."""
    tmp_file = STATE_FILE.with_suffix('.tmp')
    with open(tmp_file, 'w') as f:
        json.dump(state, f, indent=2)
    tmp_file.rename(STATE_FILE)


def is_article_read(state, entry):
    """Check if an article is marked as read."""
    source = get_source_from_url(entry['link'])
    article_id = get_article_id(entry['link'])
    sources = state.get("articles", {}).get("sources", {})
    return sources.get(source, {}).get(article_id, {}).get("markedRead", False)


def toggle_article_read(state, entry):
    """Toggle an article's read status."""
    source = get_source_from_url(entry['link'])
    article_id = get_article_id(entry['link'])
    if "articles" not in state:
        state["articles"] = {"sources": {}}
    if "sources" not in state["articles"]:
        state["articles"]["sources"] = {}
    if source not in state["articles"]["sources"]:
        state["articles"]["sources"][source] = {}
    current = state["articles"]["sources"][source].get(article_id, {}).get("markedRead", False)
    state["articles"]["sources"][source][article_id] = {"markedRead": not current}
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
            self.output.append(Color.PURP.value)
        elif tag == 'code':
            self.output.append(Color.GREEN.value)
        elif tag in ('h1', 'h2', 'h3', 'h4'):
            self.output.append(f'\n\n{BOLD}{Color.PURP.value}')
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
            self.output.append(f'\n{Color.DIM.value}')
        elif tag == 'blockquote':
            self.output.append(f'\n{Color.DIM.value}  │ ')
        elif tag == 'img':
            alt = dict(attrs).get('alt', 'image')
            self.output.append(f'{Color.DIM.value}[{alt}]{RESET}')

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
            data = ' '.join(data.split())  # normalize whitespace
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
            if line.startswith('  ') or line.startswith('│'):  # preserve indented/quoted
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
    ns = {'content': 'http://purl.org/rss/1.0/modules/content/'}

    for item in tree.findall('.//item'):
        entry = {
            'title': item.findtext('title', ''),
            'link': item.findtext('link', ''),
            'date': item.findtext('pubDate', ''),
            'description': item.findtext('description', ''),
        }
        # Try to get full content
        content = item.find('content:encoded', ns)
        if content is not None and content.text:
            entry['content'] = content.text
        else:
            entry['content'] = entry['description']
        entries.append(entry)
    return entries


def get_atom_entries(root):
    """Extract entries from Atom feed."""
    entries = []
    # Atom namespace
    ATOM = '{http://www.w3.org/2005/Atom}'

    atom_entries = root.findall(f'.//{ATOM}entry') or root.findall('.//entry')

    for item in atom_entries:
        # Get link href attribute (Atom uses <link href="..."/>)
        # Prefer rel="alternate", fall back to first link
        link = ''
        for link_elem in item.findall(f'{ATOM}link') or item.findall('link'):
            if link_elem.get('rel', 'alternate') == 'alternate':
                link = link_elem.get('href', '')
                break
        if not link:
            link_elem = item.find(f'{ATOM}link')
            if link_elem is None:
                link_elem = item.find('link')
            link = link_elem.get('href', '') if link_elem is not None else ''

        # Get content or summary
        content_elem = item.find(f'{ATOM}content')
        if content_elem is None:
            content_elem = item.find('content')
        summary_elem = item.find(f'{ATOM}summary')
        if summary_elem is None:
            summary_elem = item.find('summary')
        content = ''
        if content_elem is not None and content_elem.text:
            content = content_elem.text
        elif summary_elem is not None and summary_elem.text:
            content = summary_elem.text

        entry = {
            'title': item.findtext(f'{ATOM}title', '') or item.findtext('title', ''),
            'link': link,
            'date': item.findtext(f'{ATOM}published', '') or item.findtext(f'{ATOM}updated', '') \
                    or item.findtext('published', '') or item.findtext('updated', ''),
            'description': content,
            'content': content,
        }
        entries.append(entry)
    return entries


def get_entries(tree):
    """Extract entries from RSS or Atom feed."""
    root = tree.getroot()
    # Detect Atom feed by root tag
    if root.tag == '{http://www.w3.org/2005/Atom}feed' or root.tag == 'feed':
        return get_atom_entries(root)
    else:
        return get_rss_entries(tree)


def truncate_title(title, max_width):
    """Truncate title with ellipsis if too long."""
    if len(title) <= max_width:
        return title
    return title[:max_width - 1] + "…"


def list_entries(entries, selected=None, offset=0, height=20, selection_active=True, state=None):
    """Display list of entries with optional selection highlight."""
    term_width = shutil.get_terminal_size().columns
    print(f"{BOLD}Found {len(entries)} articles:{RESET}  {Color.DIM.value}[i/k: move, I/K: ×10, 0-9: jump, Enter: open, r: toggle read, q: quit]{RESET}{CLEAR_LINE}\n")
    for i in range(offset, min(offset + height, len(entries))):
        entry = entries[i]
        date = entry['date'].split(' +')[0] if entry['date'] else ''
        source = entry.get('source', '')
        source_str = f"[{source}] " if source else ''
        # Calculate prefix length: "  8. [source] " = 5 chars + source_str
        prefix_len = 5 + len(source_str)
        max_title_width = term_width - prefix_len - 1
        title = truncate_title(entry['title'], max_title_width)
        is_read = state and is_article_read(state, entry)
        if i == selected:
            if selection_active:
                # White/bright selection
                print(f"{BG_WHITE}{BLACK}{i+1:3}. {source_str}{title}{RESET}{CLEAR_LINE}")
                print(f"{BG_WHITE}{BLACK}     {date}{RESET}{CLEAR_LINE}")
            else:
                # Dimmed selection (gray)
                print(f"{Color.DIM.value}{REVERSE}{i+1:3}. {source_str}{title}{RESET}{CLEAR_LINE}")
                print(f"{Color.DIM.value}{REVERSE}     {date}{RESET}{CLEAR_LINE}")
        elif is_read:
            # Read articles: title in red
            print(f"{Color.PURP.value}{i+1:3}.{RESET} {Color.DIM.value}{source_str}{RESET}{Color.RED.value}{title}{RESET}{CLEAR_LINE}")
            print(f"     {Color.DIM.value}{date}{RESET}{CLEAR_LINE}")
        else:
            print(f"{Color.PURP.value}{i+1:3}.{RESET} {Color.DIM.value}{source_str}{RESET}{BOLD}{title}{RESET}{CLEAR_LINE}")
            print(f"     {Color.DIM.value}{date}{RESET}{CLEAR_LINE}")


def show_article(entry, width=80):
    """Display a single article."""
    print(f"\n{'─' * width}")
    print(f"{BOLD}{Color.PURP.value}{entry['title']}{RESET}")
    print(f"{Color.DIM.value}{entry['date']}{RESET}")
    print(f"{Color.PURP.value}{entry['link']}{RESET}")
    print(f"{'─' * width}\n")

    content = html_to_text(entry['content'], width)
    print(wrap_text(content, width))
    print(f"\n{'─' * width}")


def draw_number_bar(written_number, active, width):
    """Draw the number input bar at the bottom."""
    label = "> "
    num_str = written_number if written_number else ""
    if active:
        print(f"\n{BG_WHITE}{BLACK}{label}{num_str}_{' ' * (width - len(label) - len(num_str) - 1)}{RESET}{CLEAR_LINE}")
    else:
        print(f"\n{Color.DIM.value}{label}{num_str}{RESET}{CLEAR_LINE}")
    print(CLEAR_TO_END, end='')  # Clear any leftover lines below


def interactive_mode(entries, width=80):
    """Interactive article browser with vim-style navigation."""
    selected = 0
    term_height = shutil.get_terminal_size().lines - 6  # Leave room for header/footer/number bar
    visible_count = max(1, term_height // 2)  # 2 lines per entry
    offset = 0
    written_number = ""
    currently_writing_number = False
    state = load_state()

    print(CLEAR_SCREEN, end='')  # Initial clear only
    while True:
        # Adjust offset to keep selection visible
        if selected < offset:
            offset = selected
        elif selected >= offset + visible_count:
            offset = selected - visible_count + 1

        print(HOME, end='')
        list_entries(entries, selected, offset, visible_count, selection_active=not currently_writing_number, state=state)
        draw_number_bar(written_number, currently_writing_number, width)

        try:
            key = getch()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if key == 'q' or key == '\x03':  # q or Ctrl+C
            break
        elif key in '0123456789':
            written_number += key
            currently_writing_number = True
        elif key == '\x7f' or key == '\x08':  # Backspace
            if written_number:
                written_number = written_number[:-1]
            currently_writing_number = True  # Backspace always focuses the number bar
        elif key == 'k' or key == '\x0b':  # down (k or Ctrl+K)
            selected = min(selected + 1, len(entries) - 1)
            currently_writing_number = False
        elif key == 'i' or key == '\x1e':  # up (i or Ctrl+^ => as Ctrl+i is mapped as Ctrl+^ in my Alacritty)
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
        elif key == '\r' or key == '\n':  # Enter
            if currently_writing_number and written_number:
                target = int(written_number) - 1  # Convert to 0-indexed
                if 0 <= target < len(entries):
                    selected = target
                    print(CLEAR_SCREEN, end='')
                    show_article(entries[selected], width)
                    print(f"\n{Color.DIM.value}Press any key to continue...{RESET}")
                    getch()
                written_number = ""
                currently_writing_number = False
            else:
                print(CLEAR_SCREEN, end='')
                show_article(entries[selected], width)
                print(f"\n{Color.DIM.value}Press any key to continue...{RESET}")
                getch()


def parse_date(date_str):
    """Parse RSS (RFC 822) or Atom (ISO 8601) date string for sorting."""
    from email.utils import parsedate_to_datetime
    from datetime import datetime
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


def fetch_all_feeds():
    """Fetch all configured feeds concurrently and merge entries sorted by date."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_entries = []
    errors = []

    print(f"{Color.DIM.value}Fetching {len(FEEDS)} feeds...{RESET}")

    with ThreadPoolExecutor(max_workers=len(FEEDS)) as executor:
        futures = {
            executor.submit(fetch_single_feed, name, feed_value): name
            for name, feed_value in FEEDS.items()
        }

        for future in as_completed(futures):
            name, entries, error = future.result()
            if error:
                errors.append(f"{name}: {error}")
                print(f"{Color.RED.value}Error fetching {name}: {error}{RESET}")
            else:
                print(f"{Color.DIM.value}Fetched {name} ({len(entries)} articles){RESET}")
                all_entries.extend(entries)

    if errors:
        print(f"\n{Color.DIM.value}Press any key to continue...{RESET}")
        getch()

    from datetime import datetime
    all_entries.sort(key=lambda e: parse_date(e['date']) or datetime.min, reverse=True)
    return all_entries


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Zero-dep CLI RSS reader')
    parser.add_argument('feed', nargs='?', default=None,
                        help=f"Feed URL or shortcut: {', '.join(FEEDS.keys())} (default: all)")
    parser.add_argument('-l', '--list', action='store_true',
                        help='List articles without interactive mode')
    parser.add_argument('-n', '--number', type=int,
                        help='Show article N directly')
    parser.add_argument('-w', '--width', type=int, default=80,
                        help='Terminal width (default: 80)')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable ANSI colors')
    args = parser.parse_args()

    if args.no_color:
        global BOLD, ITALIC, BG_WHITE, BLACK, REVERSE, RESET
        BOLD = ITALIC = BG_WHITE = BLACK = REVERSE = RESET = ''
        for c in Color:
            c._value_ = ''

    if args.feed is None:
        # Fetch all feeds
        entries = fetch_all_feeds()
    else:
        # Resolve single feed URL
        feed_value = FEEDS.get(args.feed, args.feed)
        if isinstance(feed_value, str) and not feed_value.startswith('http'):
            print(f"Unknown feed: {args.feed}")
            print(f"Available shortcuts: {', '.join(FEEDS.keys())}")
            sys.exit(1)

        url, cookies = get_feed_config(feed_value)
        print(f"{Color.DIM.value}Fetching {url}...{RESET}")
        try:
            tree = fetch_feed(url, cookies)
            entries = get_entries(tree)
        except Exception as e:
            print(f"Error fetching feed: {e}")
            sys.exit(1)

    if not entries:
        print("No entries found in feed.")
        sys.exit(1)

    if args.number:
        if 1 <= args.number <= len(entries):
            show_article(entries[args.number - 1], args.width)
        else:
            print(f"Invalid article number. Choose 1-{len(entries)}")
            sys.exit(1)
    elif args.list:
        list_entries(entries, height=len(entries))
    else:
        interactive_mode(entries, args.width)


if __name__ == '__main__':
    main()
