#!/usr/bin/env python3
"""Zero-dependency CLI RSS reader with HTML rendering."""

import json
import sys
import textwrap
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen, Request
from xml.etree import ElementTree as ET

# ANSI escape codes
BOLD = '\033[1m'
ITALIC = '\033[3m'
DIM = '\033[2m'
BLUE = '\033[34m'
CYAN = '\033[36m'
YELLOW = '\033[33m'
RESET = '\033[0m'

CONFIG_DIR = Path.home() / '.config' / 'feed'
CONFIG_FILE = CONFIG_DIR / 'config.json'

EXAMPLE_FEEDS = {
    'example': 'https://example.com/feed.xml',
    'another': 'https://blog.example.org/rss',
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
            self.output.append(BLUE)
        elif tag == 'code':
            self.output.append(CYAN)
        elif tag in ('h1', 'h2', 'h3', 'h4'):
            self.output.append(f'\n\n{BOLD}{YELLOW}')
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


def fetch_feed(url):
    """Fetch and parse RSS feed."""
    # Safe: ElementTree doesn't process external entities (no XXE),
    # and Python 3.7.1+ limits entity expansion (no billion laughs)
    req = Request(url, headers={'User-Agent': 'Python RSS Reader'})
    with urlopen(req, timeout=10) as response:
        return ET.parse(response)


def get_entries(tree):
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


def list_entries(entries):
    """Display list of entries."""
    print(f"\n{BOLD}Found {len(entries)} articles:{RESET}\n")
    for i, entry in enumerate(entries, 1):
        date = entry['date'].split(' +')[0] if entry['date'] else ''
        print(f"{YELLOW}{i:2}.{RESET} {BOLD}{entry['title']}{RESET}")
        print(f"    {DIM}{date}{RESET}\n")


def show_article(entry, width=80):
    """Display a single article."""
    print(f"\n{'─' * width}")
    print(f"{BOLD}{YELLOW}{entry['title']}{RESET}")
    print(f"{DIM}{entry['date']}{RESET}")
    print(f"{BLUE}{entry['link']}{RESET}")
    print(f"{'─' * width}\n")

    content = html_to_text(entry['content'], width)
    print(wrap_text(content, width))
    print(f"\n{'─' * width}")


def interactive_mode(entries, width=80):
    """Interactive article browser."""
    while True:
        list_entries(entries)
        try:
            choice = input(f"{CYAN}Enter article number (or 'q' to quit): {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice.lower() in ('q', 'quit', 'exit'):
            break

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(entries):
                show_article(entries[idx], width)
                input(f"\n{DIM}Press Enter to continue...{RESET}")
            else:
                print(f"{YELLOW}Invalid number. Choose 1-{len(entries)}{RESET}")
        except ValueError:
            print(f"{YELLOW}Enter a number or 'q' to quit{RESET}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Zero-dep CLI RSS reader')
    parser.add_argument('feed', nargs='?', default='korben',
                        help=f"Feed URL or shortcut: {', '.join(FEEDS.keys())}")
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
        global BOLD, ITALIC, DIM, BLUE, CYAN, YELLOW, RESET
        BOLD = ITALIC = DIM = BLUE = CYAN = YELLOW = RESET = ''

    # Resolve feed URL
    url = FEEDS.get(args.feed, args.feed)
    if not url.startswith('http'):
        print(f"Unknown feed: {args.feed}")
        print(f"Available shortcuts: {', '.join(FEEDS.keys())}")
        sys.exit(1)

    print(f"{DIM}Fetching {url}...{RESET}")
    try:
        tree = fetch_feed(url)
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
        list_entries(entries)
    else:
        interactive_mode(entries, args.width)


if __name__ == '__main__':
    main()
