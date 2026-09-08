import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import _leaderboard_findings as lf


def _wellformed(fragment: str):
    # wrap so multiple top-level nodes are allowed
    ET.fromstring(f"<root>{fragment}</root>")
    assert "�" not in fragment


def _all_fragments():
    """One representative fragment per primitive (+ a few option variants)."""
    return [
        lf.svg_heatmap(["a", "b"], ["x", "y"], [[1, 2], [3, 4]]),
        lf.svg_heatmap(["a"], ["x"], [[1]], rtl_labels=True, title="t & <b>"),
        lf.hbar([("alef", 5), ("baa", 3)]),
        lf.stacked_hbar([("m1", {"sub": 3, "ins": 1, "del": 2})], ["sub", "ins", "del"]),
        lf.stacked_hbar(
            [("m1", {"a": 3, "b": 1, "c": 2, "d": 1, "e": 4}),
             ("m2", {"a": 0, "b": 0, "c": 1, "d": 0, "e": 0})],
            ["a", "b", "c", "d", "e"], normalize=True,
        ),
        lf.stacked_hbar([("m1", {"a": 3, "b": 1, "c": 2})], ["a", "b", "c"],
                        colors=["--fb-seg-3", "--fb-seg-1", 5]),
        lf.diverging_hbar([("fatha", {"missed": 4, "inserted": 1})],
                          left_key="missed", right_key="inserted",
                          left_label="missed", right_label="inserted"),
        lf.svg_scatter([(1.0, 0.5), (2.0, 0.3)], x_label="$", y_label="CER", pareto=True),
        lf.svg_scatter([(0.0, 0.6), (1.0, 0.4), (4.0, 0.2)], x_label="$", y_label="CER",
                       pareto=True, labels=["free", "a", "b"]),
        lf.svg_scatter([(0.0, 0.6), (10.0, 0.2)], x_label="$", y_label="CER", log_x=True),
        lf.binned_curve({"m1": [(5, 0.2, 10), (15, 0.4, 8)]}, x_label="len", y_label="CER"),
        lf.histogram([0.1, 0.2, 0.2, 0.9]),
        lf.ecdf({"m1": [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]}),
        lf.box_row([("m1", {"min": 0, "q1": 0.1, "median": 0.2, "q3": 0.4, "max": 1.0})]),
        lf.dumbbell([("m1", 0.2, 0.5)], left_label="rasm", right_label="light"),
        lf.forest([("m1", 0.3, 0.25, 0.36)]),
    ]


def test_every_primitive_wellformed_and_nonempty():
    for frag in _all_fragments():
        _wellformed(frag)


def test_primitives_handle_empty_input():
    for frag in (lf.hbar([]), lf.svg_scatter([], x_label="x", y_label="y"),
                 lf.box_row([]), lf.forest([]), lf.binned_curve({}, x_label="x", y_label="y"),
                 lf.svg_heatmap([], [], []), lf.stacked_hbar([], []),
                 lf.diverging_hbar([], left_key="a", right_key="b", left_label="a", right_label="b"),
                 lf.histogram([]), lf.ecdf({}), lf.dumbbell([], left_label="a", right_label="b")):
        _wellformed(frag)
        assert 'class="chart-empty"' in frag


def test_primitives_escape_markup_in_labels():
    frag = lf.hbar([("<b>a&b</b>", 3), ("q\"t", 1)])
    _wellformed(frag)
    assert "<b>a&b</b>" not in frag
    frag = lf.svg_heatmap(["<r>"], ["<c>"], [[1]])
    _wellformed(frag)


def test_no_primitive_emits_a_script_tag():
    for frag in _all_fragments():
        assert "<script" not in frag.lower()


def test_stacked_hbar_5_segments_use_distinct_colour_vars():
    frag = lf.stacked_hbar(
        [("m1", {"a": 2, "b": 2, "c": 2, "d": 2, "e": 2})],
        ["a", "b", "c", "d", "e"],
    )
    _wellformed(frag)
    for n in range(1, 6):
        assert f"var(--fb-seg-{n})" in frag
    assert "opacity:" not in frag  # flat fill per segment, no opacity-step scheme


def _fixture_findings():
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from arabic_ocr_bench.analysis.findings import build
    return build(Path(__file__).parent / "fixtures" / "mini_results")


def test_render_findings_view_has_all_sections():
    html = lf.render_findings_view(_fixture_findings())
    _wellformed(html)
    for slug, _ in lf.FINDINGS_SECTIONS:
        assert (f"id='f-{slug}'" in html) or (f'id="f-{slug}"' in html)


def test_render_findings_view_empty_dict_is_graceful():
    html = lf.render_findings_view({})
    _wellformed(html)
    assert "not available" in html.lower()
    for slug, _ in lf.FINDINGS_SECTIONS:
        assert (f"id='f-{slug}'" in html) or (f'id="f-{slug}"' in html)


def test_no_replacement_char():
    assert "�" not in lf.render_findings_view(_fixture_findings())


def test_css_is_a_string_scoped_to_findings():
    css = lf.FINDINGS_CSS
    assert isinstance(css, str)
    assert "<style" not in css
    assert "chart-empty" in css and "chart-tip" in css
    assert "--fb-seg-1" in css and "--fb-seg-5" in css
    # every rule-opening line is scoped under .findings-view (skip at-rules /
    # :root and declaration-continuation lines that carry no "{").
    for line in css.splitlines():
        s = line.strip()
        if "{" in s and not s.startswith(("@", ":root")):
            assert ".findings-view" in s, f"unscoped rule: {s!r}"
