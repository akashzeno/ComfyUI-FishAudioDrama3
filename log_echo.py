"""Forward a job's log file to ComfyUI's terminal as it grows.

The job process writes to generation.log (kept for the error message if it fails); this copies each
new piece to the terminal, so Fish Speech's own output — loading messages and its frame-by-frame
progress bar — shows up like any other node's. Reads bytes and decodes incrementally, so a
character split across two reads is not mangled. No ComfyUI imports, so it is testable on its own.
"""
import codecs


class LogEcho:
    def __init__(self, path, write):
        self.path = path
        self.write = write
        self.offset = 0
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')

    def pump(self):
        """Forward whatever was appended since the last call."""
        try:
            with open(self.path, 'rb') as file:
                file.seek(self.offset)
                data = file.read()
        except FileNotFoundError:
            return
        self.offset += len(data)
        text = self.decoder.decode(data)
        if text:
            self.write(text)
