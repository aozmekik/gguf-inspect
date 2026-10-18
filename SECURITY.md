# Security policy

`gguf-inspect` reads files that may come from anywhere, such as model hubs,
so it treats every input as hostile. These are bugs worth reporting:

- a panic, abort, stack overflow or hang on any input, in the library or the
  CLI;
- memory use far beyond what [`Limits`](src/limits.rs) documents, or an
  allocation sized by a declared count rather than by bytes in the file;
- a control or invisible format character from the file reaching the
  terminal through the CLI's text or JSON output;
- a number (sizes, offsets, parameter counts, KV-cache estimates) that is
  wrong without an error or a warning.

## Supported versions

Only the latest release (currently 0.1.x) receives fixes.

## Reporting

For anything that could hurt users of the tool, such as a crafted file that
crashes it or gets escape sequences to a terminal, please report privately
through GitHub's private vulnerability reporting on this repository
(Security tab, "Report a vulnerability"). Other bugs can go to the public
issue tracker.

Please include the smallest input that shows the problem, or a script that
generates it, and the command or library call you ran. Do not attach real
model files; a truncated header (`head -c <bytes>`) is usually enough.
