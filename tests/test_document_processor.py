"""
test_document_processor.py
---------------------------
Unit tests for document_processor.py. These tests do NOT require:
  - A real PDF file
  - ChromaDB
  - The Groq API
  - Network access

They test pure logic: department detection and hash computation.
Fast, deterministic, no external dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import pytest
import document_processor


class TestGetDepartmentFromFilename:
    """Tests for the filename → department mapping logic."""

    def test_signal_prefix(self):
        """Files starting with signal_ should map to signal department."""
        assert document_processor.get_department_from_filename(
            "signal_axle_counter_maintenance.pdf"
        ) == "signal"

    def test_electrical_prefix(self):
        """Files starting with electrical_ should map to electrical."""
        assert document_processor.get_department_from_filename(
            "electrical_reference_manual.pdf"
        ) == "electrical"

    def test_operations_prefix(self):
        assert document_processor.get_department_from_filename(
            "operations_model_sop_2018.pdf"
        ) == "operations"

    def test_safety_prefix(self):
        assert document_processor.get_department_from_filename(
            "safety_disaster_management_plan.pdf"
        ) == "safety"

    def test_it_prefix(self):
        assert document_processor.get_department_from_filename(
            "it_server_maintenance.pdf"
        ) == "it"

    def test_announcement_prefix(self):
        assert document_processor.get_department_from_filename(
            "announcement_pa_system.pdf"
        ) == "announcement"

    def test_unknown_prefix_defaults_to_operations(self):
        """
        Files with no recognised prefix should fall back to 'operations'
        rather than crashing — graceful degradation.
        """
        assert document_processor.get_department_from_filename(
            "unknown_document.pdf"
        ) == "operations"

    def test_uppercase_prefix_is_handled(self):
        """Prefix detection is case-insensitive."""
        assert document_processor.get_department_from_filename(
            "SIGNAL_manual.pdf"
        ) == "signal"

    def test_no_underscore_falls_back(self):
        """A filename with no underscore at all should not crash."""
        result = document_processor.get_department_from_filename("manual.pdf")
        assert result in list(["signal","electrical","it",
                                "operations","announcement","safety","operations"])


class TestComputeFileHash:
    """Tests for the MD5 hash function used to detect file changes."""

    def test_same_content_same_hash(self, tmp_path):
        """
        Two files with identical content must produce the same hash —
        this is the guarantee our skip-on-rerun logic depends on.
        """
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"Railway SOP content")
        f2.write_bytes(b"Railway SOP content")
        assert document_processor.compute_file_hash(f1) == \
               document_processor.compute_file_hash(f2)

    def test_different_content_different_hash(self, tmp_path):
        """
        Two files with different content must produce different hashes —
        ensures changed PDFs are re-indexed rather than silently skipped.
        """
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"Original content")
        f2.write_bytes(b"Updated content")
        assert document_processor.compute_file_hash(f1) != \
               document_processor.compute_file_hash(f2)

    def test_hash_is_string(self, tmp_path):
        """Hash must be a string (for JSON serialisation in metadata.json)."""
        f = tmp_path / "test.pdf"
        f.write_bytes(b"test")
        result = document_processor.compute_file_hash(f)
        assert isinstance(result, str)
        assert len(result) == 32  # MD5 is always 32 hex characters