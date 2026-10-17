"""Minimal Hunspell/Aspell affix expander (an `unmunch` substitute; no hunspell binary on this machine).

Supports what the Arabic dictionaries use: FLAG long / single-char flags, AF aliases, PFX/SFX with cross product,
continuation classes on affix rules (twofold suffixes and suffix-licensed prefixes). Not supported (and not used by
these dictionaries): compounding, CIRCUMFIX, NEEDAFFIX, COMPLEXPREFIXES.

Generation per stem, mirroring Hunspell's accept rules:
  stem; stem+S1 (S1 in stem flags); stem+S1+S2 (S2 in cont(S1));
  P+any of those, where P is in stem flags (cross product needed when a stem-flag suffix is present) or in the
  continuation flags of an applied suffix; stem+P then +S where S is in cont(P).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    kind: str          # "PFX" | "SFX"
    flag: str
    cross: bool
    strip: str
    add: str
    cont: frozenset
    cond: re.Pattern | None


class AffixSet:
    def __init__(self, aff_path: str, flag_mode: str | None = None):
        self.flag_mode = flag_mode or "char"
        self.af: dict[str, frozenset] = {}
        self.rules: dict[str, list[Rule]] = {}
        heads: dict[str, tuple[str, bool]] = {}
        with open(aff_path, encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                line = raw.lstrip("﻿").rstrip("\n")
                parts = line.split()
                if not parts or parts[0].startswith("#"):
                    continue
                if parts[0] == "FLAG" and len(parts) > 1 and flag_mode is None:
                    self.flag_mode = parts[1]
                elif parts[0] == "AF" and len(parts) > 1 and not parts[1].isdigit():
                    self.af[str(len(self.af) + 1)] = frozenset(self.split_flags(parts[1]))
                elif parts[0] in ("PFX", "SFX") and len(parts) >= 4:
                    kind, flag = parts[0], parts[1]
                    if len(parts) == 4 and flag not in heads:          # header: PFX flag Y|N count
                        heads[flag] = (kind, parts[2] == "Y")
                        continue
                    if len(parts) < 5:
                        continue
                    strip = "" if parts[2] == "0" else parts[2]
                    add, _, cont = parts[3].partition("/")
                    add = "" if add == "0" else add
                    cond = parts[4]
                    rx = None
                    if cond != ".":
                        rx = re.compile(("^" + cond) if kind == "PFX" else (cond + "$"))
                    self.rules.setdefault(flag, []).append(Rule(kind, flag, heads.get(flag, (kind, True))[1], strip, add,
                                                                frozenset(self.flags(cont)) if cont else frozenset(), rx))

    def split_flags(self, s: str) -> list[str]:
        if self.flag_mode == "long":
            return [s[i:i + 2] for i in range(0, len(s), 2)]
        if self.flag_mode == "num":
            return [x for x in s.split(",") if x]
        return list(s)

    def flags(self, s: str) -> frozenset:
        """Flags field of a dic entry or rule continuation; numeric fields are AF aliases when AF is defined."""
        if self.af and s.isdigit():
            return self.af.get(s, frozenset())
        return frozenset(self.split_flags(s))

    def _apply(self, word: str, rule: Rule) -> str | None:
        if rule.kind == "SFX":
            if rule.cond is not None and not rule.cond.search(word):
                return None
            if rule.strip and not word.endswith(rule.strip):
                return None
            base = word[: len(word) - len(rule.strip)] if rule.strip else word
            return base + rule.add
        if rule.cond is not None and not rule.cond.search(word):
            return None
        if rule.strip and not word.startswith(rule.strip):
            return None
        return rule.add + word[len(rule.strip):]

    def _of(self, flags, kind):
        for f in flags:
            for r in self.rules.get(f, ()):
                if r.kind == kind:
                    yield r

    def expand(self, word: str, flags: frozenset):
        yield word
        # suffix layer: (form, applied first suffix or None, licensed prefix flags)
        layer = [(word, None, frozenset(f for f in flags))]
        for s1 in self._of(flags, "SFX"):
            w1 = self._apply(word, s1)
            if w1 is None:
                continue
            yield w1
            layer.append((w1, s1, s1.cont))
            for s2 in self._of(s1.cont, "SFX"):
                w2 = self._apply(w1, s2)
                if w2 is None:
                    continue
                yield w2
                layer.append((w2, s1, s1.cont | s2.cont))
        for form, s1, cont in layer:
            if s1 is None:
                pflags = flags
            else:
                pflags = set(cont)
                if s1.cross:
                    pflags |= flags
            for p in self._of(pflags, "PFX"):
                if s1 is not None and p.flag in flags and p.flag not in cont and not p.cross:
                    continue
                wp = self._apply(form, p)
                if wp is None:
                    continue
                yield wp
                if s1 is None:
                    for s in self._of(p.cont, "SFX"):
                        ws = self._apply(word, s)
                        if ws is not None:
                            wps = self._apply(ws, p)
                            if wps is not None:
                                yield wps


class Checker:
    """Hunspell-style membership test (analysis, not generation): full expansion of Ayaspell is ~4e8 forms.

    Accepts exactly the forms `AffixSet.expand` would generate for some dictionary stem."""

    def __init__(self, aff: AffixSet, entries):
        self.aff = aff
        self.dic: dict[str, list[frozenset]] = {}
        for word, flags in entries:
            self.dic.setdefault(word, []).append(flags)
        self.sfx: dict[str, list[Rule]] = {}
        self.pfx: dict[str, list[Rule]] = {}
        for rules in aff.rules.values():
            for r in rules:
                (self.sfx if r.kind == "SFX" else self.pfx).setdefault(r.add, []).append(r)
        self.sfx_lens = sorted({len(k) for k in self.sfx})
        self.pfx_lens = sorted({len(k) for k in self.pfx})

    def _unsfx(self, form: str):
        """Yield (base, rule) such that applying rule to base gives form."""
        for n in self.sfx_lens:
            if n > len(form):
                break
            key = form[len(form) - n:] if n else ""
            for r in self.sfx.get(key, ()):
                base = form[: len(form) - n] + r.strip
                if base and (r.cond is None or r.cond.search(base)):
                    yield base, r

    def _unpfx(self, form: str):
        for n in self.pfx_lens:
            if n > len(form):
                break
            for r in self.pfx.get(form[:n], ()):
                base = r.strip + form[n:]
                if base and (r.cond is None or r.cond.search(base)):
                    yield base, r

    def _suffix_parses(self, form: str):
        """Yield (stem, s1, s2): form = stem (+s1 (+s2))."""
        yield form, None, None
        for w1, s_outer in self._unsfx(form):
            yield w1, s_outer, None
            for w0, s_inner in self._unsfx(w1):
                if s_outer.flag in s_inner.cont:
                    yield w0, s_inner, s_outer

    def known(self, word: str) -> bool:
        options = [(word, None)] + list(self._unpfx(word))
        for rest, p in options:
            for stem, s1, s2 in self._suffix_parses(rest):
                for flags in self.dic.get(stem, ()):
                    if s1 is not None and s1.flag not in flags and not (p is not None and s1.flag in p.cont):
                        continue
                    if p is None:
                        return True
                    cont = (s1.cont if s1 else frozenset()) | (s2.cont if s2 else frozenset())
                    if p.flag in cont:
                        return True
                    if p.flag in flags and (s1 is None or s1.flag in (p.cont) or (s1.cross and p.cross)):
                        return True
                    if s1 is not None and s1.flag in p.cont:
                        return True
        return False


def parse_dic_line(line: str, aff: AffixSet):
    """Return (word, flags, morph) for a Hunspell .dic line, or None for comments/counts."""
    line = line.rstrip("\n")
    if not line or line.startswith("#") or line.startswith("\t") or line.strip().isdigit():
        return None
    head, _, morph = line.partition("\t")
    head = head.strip()
    if not head:
        return None
    # a slash escaped as "\/" is not used in Arabic dictionaries
    word, _, fl = head.partition("/")
    return word, (aff.flags(fl) if fl else frozenset()), morph.strip()
