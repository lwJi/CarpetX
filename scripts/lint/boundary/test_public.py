from __future__ import annotations

import os
import re
import tempfile
import unittest
from unittest import mock

from files import PROJECT

from . import doc, public

HEADER = """\
#include <ostream>

namespace CarpetX {

using alias_t = int;

template <class T> struct holder_t {
  constexpr explicit holder_t(const int i) : i_(i) {}
  constexpr int index() const { return i_; }
  friend constexpr bool operator==(holder_t, holder_t) = default;
  friend holder_t make(int i);
  struct inner_t {
    int a;
  };
  using word_t = int;

private:
  int i_;
  void hidden();
  struct impl;
};

enum class kind_t { one, two };
enum class many_t {
  alpha,
  beta,
};

class [[nodiscard]] guarded_t {
  int secret();

public:
  guarded_t(guarded_t &&) noexcept;
  ~guarded_t();
  void open() &;
  void open() && = delete;
  void operator()() const;
  operator int() const;
  std::optional<holder_t<int> > held(std::source_location site = std::source_location::current()) const;
};

class forward_t;

template <class T>
std::ostream &operator<<(std::ostream &os, const holder_t<T> h) {
  return os << "{" << h.index() << "}";
}

template <class T> constexpr holder_t<T>::holder_t(int i) : i_(i) {}

int free_one(holder_t<int> h);
static_assert(sizeof(int) == 4);
void body() {
  int x = 1; // a brace in a comment {
  struct local_t {
    int y;
  };
}

} // namespace CarpetX

template <> struct std::hash<CarpetX::alias_t> {};
"""
PARAM = """\
KEYWORD mode "what the thorn does"
{
  "ok"       :: "write and read back"
  "bad-one"  :: "do one bad thing; CarpetX aborts the run"
  "bad-two"  :: "do another; CarpetX aborts the run"
} "ok"
"""


def write(root, files):
    for name, text in files.items():
        path = os.path.join(root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)


def topics(*texts, title="A topic", modes=()):
    return ((title, modes, texts),)


class Declarations(unittest.TestCase):
    def test_a_header_is_read_as_types_with_public_members_and_free_functions(self):
        found = public.declarations(HEADER)
        self.assertEqual(
            found.types,
            {
                "alias_t": (),
                "holder_t": ("index", "inner_t", "word_t"),
                "kind_t": ("one", "two"),
                "many_t": ("alpha", "beta"),
                "guarded_t": ("open", "held"),
                "forward_t": (),
            },
        )
        self.assertEqual(found.functions, ("make", "free_one", "body"))
        self.assertEqual(found.names, ("alias_t", "holder_t", "kind_t", "many_t", "guarded_t", "forward_t", "make", "free_one", "body"))

    def test_the_public_headers_read_as_expected(self):
        found = public.declared()
        self.assertEqual(list(found), list(doc.PUBLIC_HEADERS))
        general, integrator = found.values()
        self.assertEqual(general.types["level_t"], ("patch", "index", "parent", "child"))
        self.assertEqual(general.types["tl_t"], ("next",))
        self.assertEqual(general.types["valid_t"], ("valid_all", "valid_any", "explanation"))
        self.assertEqual(general.types["write_t"], ("mfab",))
        self.assertEqual(general.types["Timer"], ("start", "stop", "print"))
        self.assertEqual(general.types["symmetry_t"], ("none", "interpatch", "periodic", "reflection"))
        self.assertIn("bounds_t", general.types["window_t"])
        self.assertIn("level_bounds", general.types["window_t"])
        self.assertNotIn("impl", general.types["write_t"])
        self.assertNotIn("index_t::index_t", general.functions)
        self.assertNotIn("reduction::reduction", general.functions)
        for name in ("write", "scratch", "make_valid_all", "sync", "reduce"):
            self.assertIn(name, general.functions)
        self.assertNotIn("operator<<", general.functions)
        self.assertEqual(integrator.types["rk_dense_output_t"], ("none", "ssprk3", "rk4"))
        self.assertEqual(integrator.types["rk_config_t"], ("nstages", "feeds_flux_registers", "is_integrated"))
        self.assertEqual(integrator.types["rk_step_t"], ("rhs_computed", "state_updated"))
        self.assertEqual(integrator.functions, ("rk_dense_output_stages", "register_integrator", "run_schedule_group_at", "open_rk_step"))


class Coverage(unittest.TestCase):
    def problems(self, *texts):
        with mock.patch.object(public, "TOPICS", topics(*texts)):
            return public.coverage(public.declared())

    def test_every_declaration_is_named_in_a_topic(self):
        problems = self.problems("nothing named")
        self.assertIn(f"num_patches, declared in {doc.PUBLIC_HEADERS[0]}, is named in no topic of {public.SOURCE}", problems)
        self.assertIn(f"rk_step_t, declared in {doc.PUBLIC_HEADERS[1]}, is named in no topic of {public.SOURCE}", problems)
        self.assertEqual(len(problems), sum(len(found.names) for found in public.declared().values()))

    def test_a_topic_that_names_a_member_names_them_all(self):
        [problem] = [p for p in self.problems("`window_t` has `narrow`") if "names them all" in p]
        self.assertTrue(problem.startswith(f"a topic of {public.SOURCE} names window_t::narrow but not window_t::one_level, window_t::levels, "), problem)
        self.assertTrue(problem.endswith("a topic that names a member of window_t names them all"))
        self.assertEqual([p for p in self.problems("`window_t` alone", "`narrow` elsewhere") if "names them all" in p], [])
        self.assertEqual([p for p in self.problems("`tl_t::next()`", "`rk_step_t`: `rhs_computed()` and `state_updated()`") if "names them all" in p], [])

    def test_a_name_in_a_topic_that_the_headers_do_not_declare_is_refused(self):
        problems = self.problems("`foo_t`, `window_t::nope`, `CarpetX::nope`, `std::span<const gi_t>`, `tl_t::next`, `CarpetX::sync`")
        self.assertIn(f"`foo_t` in the topic 'A topic' of {public.SOURCE} is no public name", problems)
        self.assertIn(f"`window_t::nope` in the topic 'A topic' of {public.SOURCE} is no public member of window_t", problems)
        self.assertIn(f"`CarpetX::nope` in the topic 'A topic' of {public.SOURCE} is no public name", problems)
        self.assertEqual([p for p in problems if "span" in p or "next" in p or "sync`" in p], [])

    def test_the_topics_in_the_tree_cover_the_headers(self):
        self.assertEqual(public.coverage(public.declared()), [])
        self.assertEqual(public.paths(PROJECT), [])
        self.assertEqual(public.boundary(PROJECT), [])

    def test_a_path_named_in_a_topic_exists(self):
        with mock.patch.object(public, "TOPICS", topics("`TestCarpetXAPI/src/compile_checks.cxx` and `Nowhere/src/x.cxx`, not `python3 scripts/lint check --help`")):
            self.assertEqual(public.paths(PROJECT), [f"`Nowhere/src/x.cxx` in {public.SOURCE} does not exist"])


class Boundary(unittest.TestCase):
    def test_the_headers_include_nothing_private_and_are_exported_and_used(self):
        with tempfile.TemporaryDirectory() as root:
            files = {
                doc.PUBLIC_HEADERS[0]: "#include <defs.hxx>\n#include <cctk.h>\n",
                doc.PUBLIC_HEADERS[1]: "#include <carpetx.hxx>\n",
                "CarpetX/src/hierarchy/driver.hxx": "",
                "CarpetX/src/internal.hxx": "",
                public.INTERFACE: "USES INCLUDE HEADER: carpetx.hxx\nUSES INCLUDE HEADER: carpetx_rk.hxx\nINCLUDES HEADER: carpetx.hxx IN carpetx.hxx\nINCLUDES HEADER: carpetx_rk.hxx IN carpetx_rk.hxx\n",
                f"{public.CLIENT}/src/a.cxx": "#include <carpetx.hxx>\n#include <carpetx_rk.hxx>\n",
            }
            write(root, files)
            self.assertEqual(public.boundary(root), [])
            write(root, {doc.PUBLIC_HEADERS[0]: '#include "hierarchy/driver.hxx"\n#include <internal.hxx>\n'})
            self.assertEqual(
                public.boundary(root),
                [f"{doc.PUBLIC_HEADERS[0]} includes hierarchy/driver.hxx, a private file of CarpetX/src/", f"{doc.PUBLIC_HEADERS[0]} includes internal.hxx, a private file of CarpetX/src/"],
            )
            write(root, {doc.PUBLIC_HEADERS[0]: "", public.INTERFACE: "USES INCLUDE HEADER: carpetx.hxx\n", f"{public.CLIENT}/src/a.cxx": "#include <carpetx.hxx>\n"})
            self.assertEqual(
                sorted(public.boundary(root)),
                sorted(
                    [
                        f"{public.INTERFACE} lacks the line 'INCLUDES HEADER: carpetx.hxx IN carpetx.hxx'",
                        f"{public.INTERFACE} lacks the line 'USES INCLUDE HEADER: carpetx_rk.hxx'",
                        f"{public.INTERFACE} lacks the line 'INCLUDES HEADER: carpetx_rk.hxx IN carpetx_rk.hxx'",
                        f"no source of {public.CLIENT}/src/ includes <carpetx_rk.hxx>",
                    ]
                ),
            )


class Refusals(unittest.TestCase):
    def test_every_refusal_has_its_parfile_and_one_topic(self):
        with tempfile.TemporaryDirectory() as root:
            files = {
                public.MODES: PARAM,
                f"{public.ERRORS}bad_one.par": 'TestCarpetXAPI::mode = "bad-one"\n',
                f"{public.ERRORS}bad_one.expect": "CarpetX::one: refused | and how\n",
                f"{public.ERRORS}bad_two.par": 'TestCarpetXAPI::mode = "bad-two"\n',
                f"{public.ERRORS}bad_two.expect": "CarpetX::two: refused\n",
                f"{public.ERRORS}parameter.par": 'TestCarpetXAPI::mode = "ok"\n',
                f"{public.ERRORS}parameter.expect": "a parameter\n",
                f"{public.TESTS}ok.par": "",
            }
            write(root, files)
            self.assertEqual(public.modes(root), ("ok", {"ok": "write and read back", "bad-one": "do one bad thing; CarpetX aborts the run", "bad-two": "do another; CarpetX aborts the run"}))
            with mock.patch.object(public, "TOPICS", (("One", ("bad-one",), ("a",)), ("Two", ("bad-two",), ("b",)))):
                self.assertEqual(public.refusals(root), [])
            write(root, {f"{public.ERRORS}rogue.par": 'TestCarpetXAPI::mode = "bad-one"\n', f"{public.ERRORS}bad_two.par": 'TestCarpetXAPI::mode = "bad-one"\n', f"{public.ERRORS}bad_one.expect": "\n", f"{public.TESTS}ok.par": 'TestCarpetXAPI::mode = "bad-two"\n'})
            with mock.patch.object(public, "TOPICS", (("One", ("bad-one", "ok", "nope"), ("a",)), ("Two", ("bad-one",), ("b",)))):
                problems = public.refusals(root)
            self.assertEqual(
                problems,
                [
                    f"{public.ERRORS}bad_one.par has no bad_one.expect beside it, or it is empty",
                    f'{public.ERRORS}bad_two.par does not set TestCarpetXAPI::mode = "bad-two"',
                    f'{public.ERRORS}bad_two.par sets TestCarpetXAPI::mode = "bad-one" but is not bad_one.par',
                    f'{public.ERRORS}rogue.par sets TestCarpetXAPI::mode = "bad-one" but is not bad_one.par',
                    f"{public.TESTS}ok.par does not run the default mode ok",
                    f"the topic 'One' of {public.SOURCE} names the mode ok, which is not a refusal of {public.MODES}",
                    f"the topic 'One' of {public.SOURCE} names the mode nope, which is not a refusal of {public.MODES}",
                    f"the mode bad-one is a refusal of both 'One' and 'Two' in {public.SOURCE}",
                    f"the mode bad-two of {public.MODES} is a refusal of no topic of {public.SOURCE}",
                ],
            )
            os.remove(os.path.join(root, public.ERRORS, "bad_two.par"))
            with mock.patch.object(public, "TOPICS", ()):
                self.assertIn(f"the mode bad-two of {public.MODES} has no {public.ERRORS}bad_two.par", public.refusals(root))
            write(root, {public.MODES: "INT n \"n\"\n{\n} 1\n"})
            with self.assertRaisesRegex(ValueError, f"{public.MODES} has no KEYWORD mode"):
                public.modes(root)

    def test_the_refusals_in_the_tree_have_their_parfiles(self):
        self.assertEqual(public.refusals(PROJECT), [])
        default, modes = public.modes(PROJECT)
        named = [public.parfile(mode) for _, topic_modes, _ in public.TOPICS for mode in topic_modes]
        self.assertEqual(sorted(named), sorted(public.parfile(mode) for mode in modes if mode != default))
        folder = os.path.join(PROJECT, public.ERRORS)
        self.assertLess(set(named), {name for name in os.listdir(folder) if name.endswith(".par")})


class Render(unittest.TestCase):
    def test_the_contract_is_the_declarations_then_a_topic_each_with_its_refusals(self):
        text = public.render()
        self.assertTrue(text.startswith(public.TITLE + "\n\n" + public.PREAMBLE + "\n\n" + "\n".join(public.INTRO) + "\n\n" + public.declares(public.declared()) + "\n## "))
        self.assertEqual(re.findall(r"^## .*$", text, re.MULTILINE), [public.DECLARES_HEADING] + [f"## {title}" for title, _, _ in public.TOPICS])
        refused = re.findall(rf"^{public.REFUSED}(.*)$", text, re.MULTILINE)
        self.assertEqual(len(refused), sum(1 for _, modes, _ in public.TOPICS if modes))
        self.assertEqual([re.findall(r"`(\w+\.par)`", line) for line in refused], [[public.parfile(mode) for mode in modes] for _, modes, _ in public.TOPICS if modes])
        self.assertIn("\n\nRefused: `level_missing.par`, `timelevel_missing.par` and `window_outside_traversal.par`.\n\n## ", text)
        self.assertNotIn("\n|", text)
        self.assertEqual(public.series(["a"]), "a")
        self.assertEqual(public.series(["a", "b", "c"]), "a, b and c")
        for title, _, texts in public.TOPICS:
            with self.subTest(title=title):
                self.assertIn(f"\n## {title}\n\n" + "\n\n".join(texts) + "\n\n", text)
                for paragraph in texts:
                    self.assertFalse(re.search(r"\n(?!- )", paragraph.rstrip("\n")), "a paragraph is one line, but a list is one line per item")
        self.assertNotIn("```", text)
        with mock.patch.object(public, "TOPICS", ()), self.assertRaisesRegex(ValueError, "is named in no topic"):
            public.render()

    def test_doc_writes_both_generated_files(self):
        self.assertEqual(public.GENERATED, {"internals": (doc.MAP, doc.render), "public-api": (doc.API, public.render)})
        self.assertEqual(doc.API, "agent_docs/carpetx-public-api.md")


if __name__ == "__main__":
    unittest.main()
