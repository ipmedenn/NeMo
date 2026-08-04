# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts.speaker_tasks import score_diarization
from scripts.speaker_tasks.score_diarization import align_recording_ids, read_rttm_inputs, read_rttm_manifest


def write_rttm(path, recording_id, speaker="speaker_0"):
    path.write_text(
        f"SPEAKER {recording_id} 1 0.000 1.000 <NA> <NA> {speaker} <NA> <NA>\n",
        encoding="utf-8",
    )


class TestScoreDiarizationInputs:
    @pytest.mark.unit
    def test_compound_rttm_files(self, tmp_path):
        reference_path = tmp_path / "reference.rttm"
        hypothesis_path = tmp_path / "hypothesis.rttm"
        reference_path.write_text(
            "SPEAKER session_a 1 0.000 1.000 <NA> <NA> ref_a <NA> <NA>\n"
            "SPEAKER session_b 1 0.000 1.000 <NA> <NA> ref_b <NA> <NA>\n",
            encoding="utf-8",
        )
        hypothesis_path.write_text(
            "SPEAKER session_a 1 0.000 1.000 <NA> <NA> hyp_a <NA> <NA>\n"
            "SPEAKER session_b 1 0.000 1.000 <NA> <NA> hyp_b <NA> <NA>\n",
            encoding="utf-8",
        )

        reference, hypothesis = read_rttm_inputs(reference_path, hypothesis_path)

        assert set(reference) == {"session_a", "session_b"}
        assert set(hypothesis) == {"session_a", "session_b"}

    @pytest.mark.unit
    def test_matching_per_file_rttm_directories(self, tmp_path):
        reference_dir = tmp_path / "reference"
        hypothesis_dir = tmp_path / "hypothesis"
        reference_dir.mkdir()
        hypothesis_dir.mkdir()
        for filename, recording_id in (("a.rttm", "session_a"), ("b.rttm", "session_b")):
            write_rttm(reference_dir / filename, recording_id, "ref")
            write_rttm(hypothesis_dir / filename, recording_id, "hyp")

        reference, hypothesis = read_rttm_inputs(reference_dir, hypothesis_dir)

        assert set(reference) == {"session_a", "session_b"}
        assert set(hypothesis) == {"session_a", "session_b"}

    @pytest.mark.unit
    @pytest.mark.parametrize("reference_is_directory", [False, True])
    def test_mixed_compound_file_and_directory_inputs(self, tmp_path, reference_is_directory):
        compound_path = tmp_path / "compound.rttm"
        per_file_dir = tmp_path / "per_file"
        per_file_dir.mkdir()
        compound_path.write_text(
            "SPEAKER session_a 1 0.000 1.000 <NA> <NA> speaker_a <NA> <NA>\n"
            "SPEAKER session_b 1 0.000 1.000 <NA> <NA> speaker_b <NA> <NA>\n",
            encoding="utf-8",
        )
        write_rttm(per_file_dir / "a.rttm", "session_a")
        write_rttm(per_file_dir / "b.rttm", "session_b")

        reference_path, hypothesis_path = (
            (per_file_dir, compound_path) if reference_is_directory else (compound_path, per_file_dir)
        )
        reference, hypothesis = read_rttm_inputs(reference_path, hypothesis_path)

        assert set(reference) == {"session_a", "session_b"}
        assert set(hypothesis) == {"session_a", "session_b"}

    @pytest.mark.unit
    def test_directory_filenames_must_match_exactly(self, tmp_path):
        reference_dir = tmp_path / "reference"
        hypothesis_dir = tmp_path / "hypothesis"
        reference_dir.mkdir()
        hypothesis_dir.mkdir()
        write_rttm(reference_dir / "a.rttm", "session_a")
        write_rttm(hypothesis_dir / "b.rttm", "session_a")

        with pytest.raises(
            ValueError,
            match=r"missing in hypothesis: a\.rttm; not in reference: b\.rttm",
        ):
            read_rttm_inputs(reference_dir, hypothesis_dir)

    @pytest.mark.unit
    def test_empty_per_file_hypothesis_rttm_means_no_speech(self, tmp_path):
        reference_dir = tmp_path / "reference"
        hypothesis_dir = tmp_path / "hypothesis"
        reference_dir.mkdir()
        hypothesis_dir.mkdir()
        write_rttm(reference_dir / "session_a.rttm", "session_a")
        (hypothesis_dir / "session_a.rttm").write_text("", encoding="utf-8")

        reference, hypothesis = read_rttm_inputs(reference_dir, hypothesis_dir)

        assert set(reference) == {"session_a"}
        assert hypothesis == {"session_a": []}

    @pytest.mark.unit
    def test_empty_compound_hypothesis_uses_reference_recording_ids(self, tmp_path):
        reference_path = tmp_path / "reference.rttm"
        hypothesis_path = tmp_path / "hypothesis.rttm"
        reference_path.write_text(
            "SPEAKER session_a 1 0.000 1.000 <NA> <NA> speaker_a <NA> <NA>\n"
            "SPEAKER session_b 1 0.000 1.000 <NA> <NA> speaker_b <NA> <NA>\n",
            encoding="utf-8",
        )
        hypothesis_path.write_text("", encoding="utf-8")

        reference, hypothesis = read_rttm_inputs(reference_path, hypothesis_path)

        assert set(reference) == {"session_a", "session_b"}
        assert hypothesis == {"session_a": [], "session_b": []}

    @pytest.mark.unit
    def test_missing_hypothesis_recording_is_scored_as_empty(self):
        reference = {"session_a": ["0.0 1.0 ref_a"], "session_b": ["0.0 1.0 ref_b"]}
        hypothesis = {"session_a": ["0.0 1.0 hyp_a"]}

        aligned_hypothesis, message = align_recording_ids(reference, hypothesis)

        assert aligned_hypothesis == {"session_a": ["0.0 1.0 hyp_a"], "session_b": []}
        assert "session_b" in message

    @pytest.mark.unit
    def test_extra_hypothesis_recording_is_rejected(self):
        reference = {"session_a": ["0.0 1.0 ref_a"]}
        hypothesis = {"session_a": ["0.0 1.0 hyp_a"], "session_b": ["0.0 1.0 hyp_b"]}

        with pytest.raises(ValueError, match="not in reference: session_b"):
            align_recording_ids(reference, hypothesis)

    @pytest.mark.unit
    def test_main_scores_partially_missing_hypothesis_as_complete_miss(self, tmp_path):
        reference_path = tmp_path / "reference.rttm"
        hypothesis_path = tmp_path / "hypothesis.rttm"
        reference_path.write_text(
            "SPEAKER session_a 1 0.000 1.000 <NA> <NA> ref_a <NA> <NA>\n"
            "SPEAKER session_b 1 0.000 1.000 <NA> <NA> ref_b <NA> <NA>\n",
            encoding="utf-8",
        )
        write_rttm(hypothesis_path, "session_a", "hyp_a")
        args = SimpleNamespace(reference=reference_path, hypothesis=hypothesis_path, collar=0.0)

        with (
            patch.object(score_diarization, "parse_args", return_value=args),
            patch("nemo.collections.asr.metrics.der.logging.info") as log_info,
        ):
            score_diarization.main()

        log_text = "\n".join(str(call.args[0]) for call in log_info.call_args_list)
        assert "session_b" in log_text
        assert "MISS: 0.5000" in log_text

    @pytest.mark.unit
    def test_reference_manifest_resolves_rttm_and_scoring_region(self, tmp_path):
        reference_path = tmp_path / "reference.rttm"
        manifest_path = tmp_path / "manifest.json"
        write_rttm(reference_path, "internal_rttm_id", "ref")
        manifest_path.write_text(
            json.dumps(
                {
                    "audio_filepath": "audio/session_a.wav",
                    "rttm_filepath": reference_path.name,
                    "offset": 2.0,
                    "duration": 5.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        reference, audio_rttm_map = read_rttm_manifest(manifest_path)

        assert set(reference) == {"session_a"}
        assert audio_rttm_map["session_a"]["rttm_filepath"] == str(reference_path)
        assert audio_rttm_map["session_a"]["offset"] == 2.0
        assert audio_rttm_map["session_a"]["duration"] == 5.0

    @pytest.mark.unit
    def test_main_logs_full_per_file_der_report_with_manifest_reference(self, tmp_path):
        reference_path = tmp_path / "reference.rttm"
        manifest_path = tmp_path / "manifest.json"
        hypothesis_dir = tmp_path / "hypothesis"
        hypothesis_dir.mkdir()
        write_rttm(reference_path, "session_a", "ref")
        write_rttm(hypothesis_dir / "session_a.rttm", "session_a", "hyp")
        manifest_path.write_text(
            json.dumps(
                {
                    "audio_filepath": "session_a.wav",
                    "rttm_filepath": str(reference_path),
                    "duration": 1.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        args = SimpleNamespace(reference=manifest_path, hypothesis=hypothesis_dir, collar=0.0)

        with (
            patch.object(score_diarization, "parse_args", return_value=args),
            patch("nemo.collections.asr.metrics.der.logging.info") as log_info,
        ):
            score_diarization.main()

        log_text = "\n".join(str(call.args[0]) for call in log_info.call_args_list)
        assert "session_a" in log_text
        assert "false alarm" in log_text
        assert "TOTAL" in log_text

    @pytest.mark.unit
    def test_main_accepts_hypothesis_manifest(self, tmp_path):
        reference_path = tmp_path / "reference.rttm"
        hypothesis_path = tmp_path / "hypothesis.rttm"
        hypothesis_manifest = tmp_path / "hypothesis.json"
        write_rttm(reference_path, "session_a", "ref")
        write_rttm(hypothesis_path, "internal_hypothesis_id", "hyp")
        hypothesis_manifest.write_text(
            json.dumps(
                {
                    "audio_filepath": "session_a.wav",
                    "rttm_filepath": hypothesis_path.name,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        args = SimpleNamespace(reference=reference_path, hypothesis=hypothesis_manifest, collar=0.0)

        with (
            patch.object(score_diarization, "parse_args", return_value=args),
            patch("nemo.collections.asr.metrics.der.logging.info") as log_info,
        ):
            score_diarization.main()

        log_text = "\n".join(str(call.args[0]) for call in log_info.call_args_list)
        assert "session_a" in log_text
        assert "TOTAL" in log_text
