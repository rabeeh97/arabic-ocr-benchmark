"""Smart sample selection + per-run sample export (image + GT + prediction)."""

import json

from PIL import Image

from arabic_ocr_bench import samples


def _scored(cers, dataset="muharaf_lines"):
    return [
        {"sample_id": f"{dataset}/test/{i:06d}", "model": "m", "dataset": dataset,
         "cer_light": c, "cer_raw": c, "wer_light": c, "error": None}
        for i, c in enumerate(cers)
    ]


class TestAnchors:
    def test_anchor_ids_are_first_n_of_the_split(self):
        a = samples.anchor_ids("muharaf_lines", n=3)
        assert a == ["muharaf_lines/test/000000", "muharaf_lines/test/000001",
                     "muharaf_lines/test/000002"]


class TestSelect:
    def test_best_and_worst_by_cer(self):
        scored = _scored([0.9, 0.1, 0.5, 0.8, 0.2, 0.4, 0.7])
        gt_len = {r["sample_id"]: 10 for r in scored}
        picked = samples.select_sample_ids(scored, anchors=[], gt_lengths=gt_len, k=2)
        best = {sid for sid, cats in picked.items() if "best" in cats}
        worst = {sid for sid, cats in picked.items() if "worst" in cats}
        assert best == {"muharaf_lines/test/000001", "muharaf_lines/test/000004"}   # 0.1, 0.2
        assert worst == {"muharaf_lines/test/000000", "muharaf_lines/test/000003"}  # 0.9, 0.8

    def test_anchors_present_in_the_run_are_tagged(self):
        scored = _scored([0.3, 0.4, 0.5])
        gt_len = {r["sample_id"]: 5 for r in scored}
        anchors = ["muharaf_lines/test/000000", "muharaf_lines/test/000099"]  # 099 not in run
        picked = samples.select_sample_ids(scored, anchors=anchors, gt_lengths=gt_len, k=1)
        assert "anchor" in picked["muharaf_lines/test/000000"]
        assert "muharaf_lines/test/000099" not in picked

    def test_perfect_and_longest_and_typical_tagged(self):
        scored = _scored([0.0, 0.5, 0.5, 0.9])
        gt_len = {"muharaf_lines/test/000000": 4, "muharaf_lines/test/000001": 4,
                  "muharaf_lines/test/000002": 99, "muharaf_lines/test/000003": 4}
        picked = samples.select_sample_ids(scored, anchors=[], gt_lengths=gt_len, k=1)
        assert "perfect" in picked["muharaf_lines/test/000000"]
        assert "longest" in picked["muharaf_lines/test/000002"]
        assert any("typical" in c for c in picked.values())

    def test_a_sample_can_carry_multiple_categories(self):
        scored = _scored([0.0, 0.5, 0.9])
        gt_len = {r["sample_id"]: 5 for r in scored}
        anchors = ["muharaf_lines/test/000000"]
        picked = samples.select_sample_ids(scored, anchors=anchors, gt_lengths=gt_len, k=1)
        assert set(picked["muharaf_lines/test/000000"]) >= {"anchor", "best", "perfect"}

    def test_handles_fewer_rows_than_k(self):
        scored = _scored([0.3])
        gt_len = {r["sample_id"]: 5 for r in scored}
        picked = samples.select_sample_ids(scored, anchors=[], gt_lengths=gt_len, k=3)
        assert list(picked) == ["muharaf_lines/test/000000"]


class TestImageDataUri:
    def test_produces_a_jpeg_data_uri_and_downscales_width(self):
        img = Image.new("RGB", (3000, 200), "white")
        uri = samples.image_data_uri(img, max_w=900)
        assert uri.startswith("data:image/jpeg;base64,")
        assert len(uri) < 200_000  # downscaled + compressed, not the raw 3000px


class TestExport:
    def test_writes_samples_json_with_images_gt_and_prediction(self, tmp_path):
        run = tmp_path / "20260101T000000Z"
        run.mkdir()
        (run / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"sample_id": "muharaf_lines/test/000000", "model": "m", "dataset": "muharaf_lines",
             "ground_truth": "أ", "cleaned_text": "ا", "error": None},
            {"sample_id": "muharaf_lines/test/000001", "model": "m", "dataset": "muharaf_lines",
             "ground_truth": "بب", "cleaned_text": "بب", "error": None},
        ]))
        (run / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"sample_id": "muharaf_lines/test/000000", "model": "m", "dataset": "muharaf_lines",
             "cer_light": 1.0, "cer_raw": 1.0, "wer_light": 1.0, "error": None},
            {"sample_id": "muharaf_lines/test/000001", "model": "m", "dataset": "muharaf_lines",
             "cer_light": 0.0, "cer_raw": 0.0, "wer_light": 0.0, "error": None},
        ]))

        out = samples.export_run_samples(
            run, get_image=lambda i: Image.new("RGB", (200, 40), "white"), k=1
        )
        data = json.loads(out.read_text())
        assert out.name == "samples.json"
        by_id = {s["sample_id"]: s for s in data["samples"]}
        assert by_id["muharaf_lines/test/000001"]["prediction"] == "بب"
        assert by_id["muharaf_lines/test/000001"]["ground_truth"] == "بب"
        assert by_id["muharaf_lines/test/000001"]["image"].startswith("data:image/jpeg;base64,")
        assert "perfect" in by_id["muharaf_lines/test/000001"]["categories"]
