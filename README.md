# feed
KISS CLI RSS reader, in Python

## Help
```
feed - KISS cli RSS reader with HTML rendering
──────────────────────────────────────────────
~/.config/feed/config.json  => {"feeds": {"name": "https://example.com/feed.xml",
                                          "with_cookies": {"url": "https://...",
                                                           "cookies": "session=abc"}}}
~/.config/feed/state.json   => read/unread state (auto-managed)
──────────────────────────────────────────────
- feed                      ==> fetch all feeds, interactive mode
- feed name                 ==> fetch single feed by shortcut
- feed "https://..."        ==> fetch feed by URL
- feed -l                   ==> list articles without interactive mode
- feed -n 5                 ==> show article #5 directly
- feed -w 100               ==> set terminal width to 100
- feed --no-color           ==> disable ANSI colors
──────────────────────────────────────────────
Interactive mode:
  i/k or ↑/↓                ==> move selection up/down
  I/K                       ==> move selection up/down by 10
  0-9 + Enter               ==> jump to article number
  Enter                     ==> open selected article
  r                         ==> toggle read status
  q / Esc                   ==> quit
──────────────────────────────────────────────
Supports RSS and Atom feeds
```
