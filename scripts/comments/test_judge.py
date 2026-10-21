from __future__ import annotations

import unittest

from files import Lang
from judge import Touched, Verdict, judge, judge_all
from rules import Unit


def found(verdict):
    return [("new", u.line, u.text) for u in verdict.added] + [("touched", t.first, t.last) for t in verdict.touched]


def check(before, after, lang=Lang.CXX):
    return found(judge(before, after, lang))


class NewLines(unittest.TestCase):
    def test_a_new_comment(self):
        self.assertEqual(check("int a;\n", "// why\nint a;\n"), [("new", 1, "// why")])

    def test_a_trailing_comment_on_code_that_did_not_change(self):
        self.assertEqual(check("int a;\n", "int a; // why\n"), [("new", 1, "// why")])

    def test_a_new_comment_over_changed_code_is_only_new(self):
        self.assertEqual(check("int x = 1;\n", "// a\nint x = 2;\n"), [("new", 1, "// a")])

    def test_rewording_a_one_line_comment_makes_it_new(self):
        self.assertEqual(check("// a\nint x;\n", "// b\nint x;\n"), [("new", 1, "// b")])

    def test_an_appended_line(self):
        self.assertEqual(check("// a\nint x;\n", "// a\n// b\nint x;\n"), [("new", 2, "// b")])

    def test_re_wrapping_a_comment_by_hand(self):
        self.assertEqual(check("// one two\nint x;\n", "// one\n// two\nint x;\n"),
                         [("new", 1, "// one"), ("new", 2, "// two")])

    def test_a_comment_moved_elsewhere_is_new_where_it_lands(self):
        self.assertEqual(check("// a\nint x;\n\nint y;\n", "int x;\n\n// a\nint y;\n"), [("new", 3, "// a")])

    def test_a_copy(self):
        self.assertEqual(check("// a\nint x;\n", "// a\nint x;\n// a\nint y;\n"), [("new", 3, "// a")])

    def test_a_new_file(self):
        self.assertEqual(check("", "// a\nint x;\n"), [("new", 1, "// a")])

    def test_constant_guards_and_compile_time_messages(self):
        self.assertEqual(check("int a;\n", "#if 0\nint a;\n#endif\n#warning \"x\"\n"),
                         [("new", 1, "#if 0"), ("new", 4, "#warning \"x\"")])

    def test_allowed_lines_are_never_new(self):
        self.assertEqual(check("int a;\n", "namespace N {\nint a;\n} // namespace N\n// doi:10.1000/x\nint b;\n"), [])

    def test_lines_without_words_are_never_new(self):
        self.assertEqual(check("int a;\n", "int a; //\n/* */\n"), [])


class TouchedComments(unittest.TestCase):
    def test_changing_the_code_under_a_comment(self):
        self.assertEqual(check("// a\nint x = 1;\n", "// a\nint x = 2;\n"), [("touched", 1, 1)])
        self.assertEqual(check("// a\nint x;\nint y = 1;\n", "// a\nint x;\nint y = 2;\n"), [("touched", 1, 1)])

    def test_code_past_a_blank_line_is_not_under_it(self):
        self.assertEqual(check("// a\nint x;\n\nint y = 1;\n", "// a\nint x;\n\nint y = 2;\n"), [])

    def test_inserting_code_under_a_comment(self):
        self.assertEqual(check("// a\nint y;\n", "// a\nint x;\nint y;\n"), [("touched", 1, 1)])

    def test_deleting_code_under_a_comment(self):
        self.assertEqual(check("// a\nint x;\nint y;\n", "// a\nint y;\n"), [("touched", 1, 1)])
        self.assertEqual(check("// a\nint x;\n\nint y;\n", "// a\n\nint y;\n"), [("touched", 1, 1)])

    def test_changing_a_function_under_its_header_comment(self):
        before = "// f\nvoid f() {\n  x = 1;\n\n  y = 1;\n}\n"
        self.assertEqual(check(before, before.replace("y = 1", "y = 2")), [("touched", 1, 1)])

    def test_the_code_on_a_trailing_comment_s_line(self):
        self.assertEqual(check("int x = 1; // a\n", "int x = 2; // a\n"), [("touched", 1, 1)])
        self.assertEqual(check("int x = 1; // a\n           // b\n", "int x = 2; // a\n           // b\n"),
                         [("touched", 1, 2)])

    def test_deleting_some_lines_of_a_comment(self):
        self.assertEqual(check("// a\n// b\nint x;\n", "// a\nint x;\n"), [("touched", 1, 1)])
        self.assertEqual(check("/* one\n   two\n   three */\nint x;\n", "/* one\n   three */\nint x;\n"),
                         [("touched", 1, 2)])

    def test_rewording_a_line_of_a_longer_comment(self):
        self.assertEqual(check("// a\n// b\nint x;\n", "// a\n// c\nint x;\n"), [("touched", 1, 2)])

    def test_reordering_the_lines_of_a_comment(self):
        self.assertEqual(check("// a\n// b\nint x;\n", "// b\n// a\nint x;\n"), [("touched", 1, 2)])

    def test_merging_two_comments_by_deleting_the_code_between(self):
        self.assertEqual(check("// a\nint x;\n// b\nint y;\n", "// a\n// b\nint y;\n"), [("touched", 1, 2)])

    def test_deleting_the_next_comment_does_not_widen_the_one_above(self):
        before = "// a\nint x;\n// b\nint y = 1;\n"
        self.assertEqual(check(before, "// a\nint x;\nint y = 2;\n"), [])
        before = "// f\nvoid f(int a,\n       // b\n       int b) {\n  x = 1;\n}\n"
        self.assertEqual(check(before, "// f\nvoid f(int a,\n       int b) {\n  x = 2;\n}\n"), [])

    def test_code_appended_to_the_paragraph_under_a_comment(self):
        self.assertEqual(check("// a\nint x;\n\nint y;\n", "// a\nint x;\nint z;\n\nint y;\n"), [("touched", 1, 1)])
        self.assertEqual(check("// a\nint x;\n\nint y;\n", "// a\nint x;\n\nint z;\nint y;\n"), [])

    def test_code_inserted_inside_the_code_under_a_comment_wherever_it_lands(self):
        before = "// a\nint x;\nint y;\n"
        self.assertEqual(check(before, "// a\nint x;\n\nint z;\nint y;\n"), [("touched", 1, 1)])
        self.assertEqual(check(before, "// a\nint x;\n\nint z;\n\nint y;\n"), [("touched", 1, 1)])
        self.assertEqual(check(before, "// a\nint x;\n//\nint z;\nint y;\n"), [("touched", 1, 1)])
        self.assertEqual(check(before, "// a\nint x;\n// doi:10.1000/x\nint z;\nint y;\n"), [("touched", 1, 1)])

    def test_a_blank_line_inside_the_code_under_a_comment_changes_no_code(self):
        self.assertEqual(check("// a\nint x;\nint y;\n", "// a\nint x;\n\nint y;\n"), [])

    def test_a_function_added_after_the_block_under_a_comment(self):
        before = "// a\nint f() {\n  return 1;\n}\n\nint h1;\nint h2;\nint h3;\nint h4;\n"
        after = before.replace("}\n", "}\n\nint g() {\n  return 2;\n}\n", 1)
        self.assertEqual(check(before, after), [])

    def test_code_added_above_a_comment_is_not_under_it(self):
        self.assertEqual(check("// a\nint x;\n", "int w;\n// a\nint x;\n"), [])

    def test_deleting_a_whole_comment(self):
        self.assertEqual(check("// a\n// b\nint x;\n", "int x;\n"), [])
        self.assertEqual(check("// a\n// b\nint x;\n", "int y;\n"), [])

    def test_spacing_openers_and_wrapping_code_in_a_block(self):
        self.assertEqual(check("  // a\n  int x = 1;\n", "/// a\nint  x=1;\n"), [])
        self.assertEqual(check("// one\n// two\nint x;\n", "{\n  // one\n  // two\n  int x;\n}\n"), [])

    def test_deleting_a_compile_time_message_changes_no_code(self):
        before = "// f\nvoid f() {\n#warning \"x\"\n  y;\n}\n"
        self.assertEqual(check(before, "// f\nvoid f() {\n  y;\n}\n"), [])

    def test_a_comment_of_allowed_lines_only(self):
        self.assertEqual(check("// doi:10.1000/x\nint x = 1;\n", "// doi:10.1000/x\nint x = 2;\n"), [])

    def test_a_touched_comment_reports_its_first_line_outside_the_allow_list(self):
        verdict = judge("// doi:10.1000/x\n// the x\nint x = 1;\n", "// doi:10.1000/x\n// the x\nint x = 2;\n", Lang.CXX)
        self.assertEqual([(t.first, t.last, t.unit.text) for t in verdict.touched], [(1, 2, "// the x")])

    def test_hash_languages(self):
        self.assertEqual(check("# a\nT::x = 1\n", "# a\nT::x = 2\n", Lang.PAR), [("touched", 1, 1)])
        self.assertEqual(check("! a\nx = 1\n", "! a\nx = 2\n", Lang.FORTRAN), [("touched", 1, 1)])
        param = "# a\nINT n \"n\"\n{\n  *:* :: \"\"\n} 1\n"
        self.assertEqual(check(param, param.replace("} 1", "} 2"), Lang.PARAM), [("touched", 1, 1)])
        self.assertEqual(check(param, param.replace("} 1", "} 1 # why"), Lang.PARAM), [("new", 5, "# why")])


class EveryStartingTree(unittest.TestCase):
    def test_a_comment_another_starting_tree_has(self):
        after = "// up\nint x;\n"
        self.assertEqual(found(judge_all(["int x;\n", after], after, Lang.CXX)), [])
        self.assertEqual(found(judge_all(["int x;\n", "int x;\n"], after, Lang.CXX)), [("new", 1, "// up")])

    def test_code_another_starting_tree_changed(self):
        after = "// a\nint x = 2;\n"
        self.assertEqual(found(judge_all(["// a\nint x = 1;\n", after], after, Lang.CXX)), [])

    def test_the_first_tree_names_what_is_reported(self):
        verdict = judge_all(["// a\nint x = 1;\n", "int x = 1;\n"], "// a\nint x = 2;\n", Lang.CXX)
        self.assertEqual(found(verdict), [("touched", 1, 1)])

    def test_within_keeps_what_the_other_verdict_flags_on_the_same_lines(self):
        u = Unit(3, ("comment", "x"), "// x", False)
        mine = Verdict((u,), (Touched(5, 7, u),))
        self.assertEqual(mine.within(Verdict((Unit(6, ("comment", "y"), "// y", False),))), Verdict((), (Touched(5, 7, u),)))
        self.assertEqual(mine.within(Verdict((), (Touched(1, 3, u),))), Verdict((u,), ()))
        self.assertFalse(mine.within(Verdict()))


if __name__ == "__main__":
    unittest.main()
