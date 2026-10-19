import random
import shutil
import unittest

from files import Lang
from lex_hash import lex
from strip import strip
from verify import verify_file

HAVE_CXX_TOOLS = shutil.which("clang++") is not None and shutil.which("unifdef") is not None

FOLDABLE = [
    "0", "1", "defined A", "defined(B)", "defined ( C )", "__has_include(<a.h>)", '__has_include("b.h")',
    "__has_include( <c/d.h> )", "__has_include_next(<e.h>)", '__has_include("a(b.h")',
]
FOREIGN = [
    "A", "true", "false", "0x0", "FOO(1)", "'a'", "'('", "A > 1", "2", "__has_include(<a(b.h>)",
    "__has_include(<a)b.h>)", "defined", "1'0",
]
OPERATORS = ["&&"] * 8 + ["||", "or", "and", "==", ","]


def condition(rng: random.Random, depth: int = 0) -> str:
    r = rng.random()
    if depth > 3 or r < 0.4:
        return rng.choice(FOREIGN if rng.random() < 0.08 else FOLDABLE)
    if r < 0.55:
        return "(" + condition(rng, depth + 1) + ")"
    if r < 0.65:
        return rng.choice(["!", "! ", "!!"]) + condition(rng, depth + 1)
    left, right = condition(rng, depth + 1), condition(rng, depth + 1)
    return left + rng.choice([" ", ""]) + rng.choice(OPERATORS) + " " + right


def guard(k: int, line: str) -> str:
    return f"{line}\nx{k}();\n#else\ny{k}();\n#endif\nint z{k};\n"


def segments(text: str, count: int) -> list[str]:
    found, start = [], 0
    for k in range(count):
        end = text.index(f"int z{k};\n", start) + len(f"int z{k};\n")
        found.append(text[start:end])
        start = end
    return found


@unittest.skipUnless(HAVE_CXX_TOOLS, "needs clang++ and unifdef")
class GeneratedGuards(unittest.TestCase):
    def test_every_guard_strip_resolves_unifdef_resolves_the_same_way(self):
        rng = random.Random(1)
        lines = sorted({rng.choice(["#if ", "# if ", "  #  if "]) + condition(rng) for _ in range(4000)})
        before = [guard(k, line) for k, line in enumerate(lines)]
        after = segments(strip("".join(before), Lang.CXX), len(lines))
        changed = [line for line, b, a in zip(lines, before, after) if a != b]
        text = "".join(guard(k, line) for k, line in enumerate(changed))
        result = verify_file("X/src/a.cxx", text, strip(text, Lang.CXX), None)
        self.assertTrue(result.ok, result.detail)
        self.assertGreater(len(changed), 100)


CONTINUED = [
    "s = 'a !b&", "&c ! d'", "&e&", "t = 1 ! y", "! note", 'x = "p&', '&q" ! r', "", "&f' // 'g ! h&",
    "u = 'v' ! w", "&k&", "  &m ! n'", "z = 'p&", "  q ! r", "call g(1) ! c",
]
CPP_OPENERS = ["#ifdef A", "#if B", "#ifndef C", "# if defined(D)"]


def program(rng: random.Random, depth: int = 0) -> list:
    items = []
    for _ in range(rng.randint(1, 4)):
        if depth < 3 and rng.random() < 0.35:
            n = rng.randint(1, 3)
            items.append(([program(rng, depth + 1) for _ in range(n)], n > 1 and rng.random() < 0.6))
        else:
            items.append(rng.choice(CONTINUED))
    return items


def render(rng: random.Random, items: list, lines: list[str]) -> list[list[int]]:
    readings = [[]]
    for item in items:
        if isinstance(item, str):
            lines.append(item)
            readings = [r + [len(lines) - 1] for r in readings]
            continue
        branches, has_else = item
        lines.append(rng.choice(CPP_OPENERS))
        options = []
        for k, branch in enumerate(branches):
            if k:
                lines.append("#else" if has_else and k == len(branches) - 1 else "#elif E")
            options += render(rng, branch, lines)
        lines.append("#endif")
        if not has_else:
            options.append([])
        readings = [r + o for r in readings for o in options][:64]
    return readings


def comment_starts(text: str) -> set[tuple[int, int, str]]:
    starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
    return {(s.line - 1, s.start - starts[s.line - 1], s.text) for s in lex(text, Lang.FORTRAN)}


class GeneratedFortranBranches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = random.Random(1)
        cls.cases = []
        for _ in range(150):
            lines: list[str] = []
            readings = render(rng, program(rng), lines)
            text = "\n".join(lines) + "\n"
            per_reading = []
            for r in readings:
                found = comment_starts("\n".join(lines[i] for i in r) + "\n")
                per_reading.append((set(r), {(r[k], col, t) for k, col, t in found}))
            cls.cases.append((lines, text, per_reading))

    def readings_of(self, per_reading, line):
        return [comments for lines, comments in per_reading if line in lines]

    def test_a_comment_is_a_comment_in_every_reading_of_the_branches(self):
        for lines, text, per_reading in self.cases:
            for span in comment_starts(text):
                for comments in self.readings_of(per_reading, span[0]):
                    self.assertIn(span, comments, text)

    def test_our_strip_passes_verify(self):
        for lines, text, per_reading in self.cases:
            result = verify_file("X/src/a.F90", text, strip(text, Lang.FORTRAN), None)
            self.assertTrue(result.ok, f"{result.detail}\n{text}")

    def test_cutting_where_some_reading_is_inside_a_string_fails(self):
        for lines, text, per_reading in self.cases:
            for number, line in enumerate(lines):
                if line.startswith("#"):
                    continue
                for col, c in enumerate(line):
                    if c != "!" or not line[:col].strip():
                        continue
                    readings = self.readings_of(per_reading, number)
                    if all(any(s[:2] == (number, col) for s in comments) for comments in readings):
                        continue
                    cut = lines[:number] + [line[:col].rstrip()] + lines[number + 1 :]
                    result = verify_file("X/src/a.F90", text, "\n".join(cut) + "\n", None)
                    self.assertFalse(result.ok, f"cut at line {number + 1} column {col + 1}\n{text}")


if __name__ == "__main__":
    unittest.main()
