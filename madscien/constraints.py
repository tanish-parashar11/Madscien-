"""
Constrained decoding for Python-like code (stdlib only; torch is imported lazily).

DelimiterConstraint tracks, byte by byte, the nesting of ( ) [ ] { }, string literals and comments
and masks every next byte that would put the output into a state that can never be balanced:
  * a closing bracket that does not match the innermost open one (or when nothing is open)
  * EOS while a bracket or string is still open
Guarantee: the generated text never has a mismatched bracket.
It does NOT guarantee that the whole program parses or is correct (that is what verify.py is for).
Known simplification: triple-quoted strings containing a bare quote character may confuse the tracker.
"""
OPEN = {"(": ")", "[": "]", "{": "}"}
CLOSE = frozenset(OPEN.values())


class DelimiterConstraint:
    def __init__(self):
        self.stack = []          # expected closers, innermost last
        self.quote = None
        self.esc = False
        self.comment = False
        self.violations = 0      # bytes fed that the mask would have forbidden

    # ---- state update
    def feed(self, tok):
        if tok > 255:
            return
        c = chr(tok)
        if self.comment:
            if c == "\n":
                self.comment = False
            return
        if self.quote:
            if self.esc:
                self.esc = False
            elif c == "\\":
                self.esc = True
            elif c == self.quote:
                self.quote = None
            return
        if c == "#":
            self.comment = True
        elif c in "\"'":
            self.quote = c
        elif c in OPEN:
            self.stack.append(OPEN[c])
        elif c in CLOSE:
            if self.stack and self.stack[-1] == c:
                self.stack.pop()
            else:
                self.violations += 1

    def feed_many(self, toks):
        for t in toks:
            self.feed(t)

    # ---- masks
    def allowed_bytes(self):
        allowed = [True] * 256
        if not self.comment and not self.quote:
            for ch in CLOSE:
                if not self.stack or self.stack[-1] != ch:
                    allowed[ord(ch)] = False
        return allowed

    def can_end(self):
        return not self.stack and not self.quote

    def allowed_mask(self, vocab_size, eos_id=None, device="cpu"):
        import torch
        m = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        n = min(256, vocab_size)
        m[:n] = torch.tensor(self.allowed_bytes()[:n], dtype=torch.bool, device=device)
        if eos_id is not None and self.can_end():
            m[eos_id] = True
        return m

    def repair(self):
        """Characters that close everything still open (use on truncated generations)."""
        s = ""
        if self.comment:
            s += "\n"
        elif self.quote:
            if self.esc:
                s += "n"
            s += self.quote
        return s + "".join(reversed(self.stack))
