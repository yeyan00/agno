"""Tests for SkillToolkit (Pi-style progressive disclosure skills)."""

import os
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

import pytest

# Ensure agno is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agno.tools.skill_toolkit import SkillToolkit, _discover_skills, _parse_frontmatter, _SkillMeta

# ---------------------------------------------------------------------------
# Fixtures: temp skill directories
# ---------------------------------------------------------------------------


@pytest.fixture
def skills_dir(tmp_path):
    """Create a temp directory with several skill folders."""

    # pdf-tools: normal skill
    pdf = tmp_path / "pdf-tools"
    pdf.mkdir()
    (pdf / "SKILL.md").write_text(
        dedent("""\
        ---
        name: pdf-tools
        description: Extract text and tables from PDF files.
        ---
        # PDF Tools
        Run `python scripts/extract.py input.pdf`.
        """),
        encoding="utf-8",
    )
    (pdf / "scripts").mkdir()
    (pdf / "scripts" / "extract.py").write_text("print('extract')", encoding="utf-8")
    (pdf / "assets").mkdir()
    (pdf / "assets" / "template.json").write_text("{}", encoding="utf-8")

    # data-analysis: normal skill
    da = tmp_path / "data-analysis"
    da.mkdir()
    (da / "SKILL.md").write_text(
        dedent("""\
        ---
        name: data-analysis
        description: Analyze datasets with pandas.
        ---
        # Data Analysis
        """),
        encoding="utf-8",
    )
    (da / "templates").mkdir()
    (da / "templates" / "report.ipynb").write_text("{}", encoding="utf-8")

    # no-desc: should be skipped
    nd = tmp_path / "no-desc"
    nd.mkdir()
    (nd / "SKILL.md").write_text(
        dedent("""\
        ---
        name: no-desc
        ---
        # No Description
        """),
        encoding="utf-8",
    )

    # hidden: disable-model-invocation
    hs = tmp_path / "hidden-skill"
    hs.mkdir()
    (hs / "SKILL.md").write_text(
        dedent("""\
        ---
        name: hidden-skill
        description: Hidden from prompt.
        disable-model-invocation: true
        ---
        # Hidden
        """),
        encoding="utf-8",
    )

    return tmp_path


# ---------------------------------------------------------------------------
# _parse_frontmatter tests
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_basic_frontmatter(self):
        content = dedent("""\
        ---
        name: my-skill
        description: Does stuff.
        ---
        # Body here
        """)
        fm = _parse_frontmatter(content)
        assert fm["name"] == "my-skill"
        assert fm["description"] == "Does stuff."

    def test_no_frontmatter(self):
        content = "# Just a markdown file\nNo frontmatter."
        fm = _parse_frontmatter(content)
        assert fm == {}

    def test_empty_frontmatter(self):
        content = "---\n---\n# Body"
        fm = _parse_frontmatter(content)
        # yaml.safe_load("---\n---") returns None
        assert fm == {}

    def test_complex_frontmatter(self):
        content = dedent("""\
        ---
        name: complex
        description: A complex skill.
        metadata:
          version: "1.0"
          tags:
            - pdf
            - text
        ---
        Body
        """)
        fm = _parse_frontmatter(content)
        assert fm["name"] == "complex"
        assert fm["metadata"]["version"] == "1.0"
        assert "pdf" in fm["metadata"]["tags"]


# ---------------------------------------------------------------------------
# _discover_skills tests
# ---------------------------------------------------------------------------


class TestDiscoverSkills:
    def test_discovers_multiple_skills(self, skills_dir):
        skills = _discover_skills(skills_dir)
        names = [s.name for s in skills]
        assert "pdf-tools" in names
        assert "data-analysis" in names

    def test_skips_no_description(self, skills_dir):
        skills = _discover_skills(skills_dir)
        names = [s.name for s in skills]
        assert "no-desc" not in names

    def test_includes_hidden_skill(self, skills_dir):
        # _discover_skills doesn't filter by disable_model_invocation
        skills = _discover_skills(skills_dir)
        names = [s.name for s in skills]
        assert "hidden-skill" in names

    def test_single_skill_folder(self, skills_dir):
        pdf_dir = skills_dir / "pdf-tools"
        skills = _discover_skills(pdf_dir)
        assert len(skills) == 1
        assert skills[0].name == "pdf-tools"

    def test_nonexistent_dir(self):
        skills = _discover_skills(Path("/nonexistent/path"))
        assert skills == []


# ---------------------------------------------------------------------------
# SkillToolkit tests
# ---------------------------------------------------------------------------


class TestSkillToolkit:
    def test_load_skills(self, skills_dir):
        tk = SkillToolkit(dirs=[skills_dir])
        names = tk.get_skill_names()
        assert "pdf-tools" in names
        assert "data-analysis" in names
        assert "hidden-skill" in names
        # no-desc skipped
        assert "no-desc" not in names

    def test_skill_meta_fields(self, skills_dir):
        tk = SkillToolkit(dirs=[skills_dir])
        pdf = tk.get_skill("pdf-tools")
        assert pdf is not None
        assert pdf.name == "pdf-tools"
        assert "PDF" in pdf.description
        assert Path(pdf.skill_file).name == "SKILL.md"
        assert Path(pdf.skill_dir).name == "pdf-tools"

    def test_instructions_contain_skills(self, skills_dir):
        tk = SkillToolkit(dirs=[skills_dir])
        instr = tk.instructions
        assert "<skills>" in instr
        assert "<available_skills>" in instr
        assert "pdf-tools" in instr
        assert "data-analysis" in instr
        assert "read_file" in instr

    def test_instructions_exclude_hidden(self, skills_dir):
        tk = SkillToolkit(dirs=[skills_dir])
        instr = tk.instructions
        assert "hidden-skill" not in instr

    def test_instructions_contain_skill_dir(self, skills_dir):
        tk = SkillToolkit(dirs=[skills_dir])
        instr = tk.instructions
        # skill_dir should be an absolute path
        assert "<skill_dir>" in instr
        assert "pdf-tools" in instr

    def test_zero_tools_registered(self, skills_dir):
        tk = SkillToolkit(dirs=[skills_dir])
        assert len(tk.functions) == 0
        assert len(tk.async_functions) == 0

    def test_add_instructions_is_true(self, skills_dir):
        tk = SkillToolkit(dirs=[skills_dir])
        assert tk.add_instructions is True

    def test_toolkit_name(self, skills_dir):
        tk = SkillToolkit(dirs=[skills_dir])
        assert tk.name == "skills"

    def test_duplicate_skill_names(self, skills_dir):
        """First skill wins when same name appears in multiple dirs."""
        # Create another dir with same-named skill
        dup_dir = skills_dir / "duplicates"
        dup_dir.mkdir()
        sub = dup_dir / "pdf-tools"
        sub.mkdir()
        (sub / "SKILL.md").write_text(
            dedent("""\
            ---
            name: pdf-tools
            description: Duplicate PDF tools.
            ---
            """),
            encoding="utf-8",
        )
        tk = SkillToolkit(dirs=[skills_dir, dup_dir])
        pdf = tk.get_skill("pdf-tools")
        # First one wins
        assert "Extract text" in pdf.description

    def test_nonexistent_dir_warns(self, skills_dir):
        """Non-existent dir should not raise, just warn."""
        tk = SkillToolkit(dirs=["/nonexistent/path"])
        assert tk.get_skill_names() == []

    def test_empty_instructions_when_no_skills(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        tk = SkillToolkit(dirs=[empty_dir])
        assert tk.instructions == ""

    def test_single_dir_string(self, skills_dir):
        """Accept a single string path."""
        tk = SkillToolkit(dirs=str(skills_dir))
        assert "pdf-tools" in tk.get_skill_names()

    def test_xml_escaping_in_description(self, tmp_path):
        skill_dir = tmp_path / "xml-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            dedent("""\
            ---
            name: xml-skill
            description: "Handles <tags> & 'quotes' and \"double quotes\""
            ---
            """),
            encoding="utf-8",
        )
        tk = SkillToolkit(dirs=[tmp_path])
        instr = tk.instructions
        assert "&lt;" in instr
        assert "&amp;" in instr
        assert "&apos;" in instr
        assert "&quot;" in instr

    def test_skill_name_from_directory_when_missing(self, tmp_path):
        """If frontmatter has no name, use directory name."""
        skill_dir = tmp_path / "auto-named"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            dedent("""\
            ---
            description: Auto-named skill.
            ---
            """),
            encoding="utf-8",
        )
        tk = SkillToolkit(dirs=[tmp_path])
        assert "auto-named" in tk.get_skill_names()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
