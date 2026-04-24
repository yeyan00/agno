"""Pi-style SkillToolkit: injects skill metadata into system prompt as instructions.

Unlike the classic ``Skills`` class (in ``agno.skills``) which registers 3 dedicated
function tools, SkillToolkit follows the **progressive disclosure** model inspired
by the Pi coding agent:

1. At init, scan skill directories and extract only ``name`` + ``description`` from frontmatter.
2. Inject an XML-formatted ``<available_skills>`` block into the system prompt via ``Toolkit.instructions``.
3. The agent uses its existing tools (e.g. ``CodingTools.read_file`` / ``CodingTools.run_shell``)
   to load SKILL.md content on-demand, access custom subdirectories (assets/, templates/, etc.),
   and execute scripts of any language.

This design:
- Adds **zero tools** to the agent.
- Naturally supports **any subdirectory** (not just scripts/ and references/).
- Supports **any scripting language** (via the agent's shell tool).
- Is fully self-contained: no changes to ``_tools.py`` or ``_messages.py`` required.

Usage::

    from agno.tools.skill_toolkit import SkillToolkit
    from agno.tools.coding import CodingTools

    agent = Agent(
        tools=[
            CodingTools(),                           # provides read_file + run_shell
            SkillToolkit(dirs=["./skills"]),          # injects skill instructions
        ]
    )

The generated system prompt instructions look like::

    <skills>
    The following skills provide specialized instructions for specific tasks.
    Use read_file to load the skill's SKILL.md file when the task matches its description.
    When a skill file references a relative path, resolve it against the skill directory.

    <available_skills>
      <skill>
        <name>pdf-tools</name>
        <description>Extract text and tables from PDF files...</description>
        <location>/abs/path/pdf-tools/SKILL.md</location>
        <skill_dir>/abs/path/pdf-tools</skill_dir>
      </skill>
    </available_skills>
    </skills>
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from agno.tools.toolkit import Toolkit
from agno.utils.log import log_debug, log_warning

# ---------------------------------------------------------------------------
# Internal skill metadata (only what we need for prompt injection)
# ---------------------------------------------------------------------------


@dataclass
class _SkillMeta:
    """Lightweight metadata extracted from a SKILL.md frontmatter."""

    name: str
    description: str
    skill_file: str  # absolute path to SKILL.md
    skill_dir: str  # absolute path to the skill folder
    disable_model_invocation: bool = False


# ---------------------------------------------------------------------------
# YAML frontmatter parsing
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


def _parse_frontmatter(content: str) -> Dict[str, Any]:
    """Extract YAML frontmatter from SKILL.md content.

    Returns an empty dict if there is no frontmatter or parsing fails.
    The body is intentionally discarded — the agent reads it on-demand.
    """
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return {}

    text = m.group(1)

    # Try yaml first (handles complex structures).
    try:
        import yaml

        result = yaml.safe_load(text)
        return result if isinstance(result, dict) else {}
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: simple key: value pairs.
    meta: Dict[str, Any] = {}
    for line in text.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip("\"'")
    return meta


# ---------------------------------------------------------------------------
# Skill discovery
# ---------------------------------------------------------------------------


def _load_skill_from_folder(folder: Path) -> Optional[_SkillMeta]:
    """Load a single _SkillMeta from a folder containing SKILL.md.

    Returns None if the folder is not a valid skill or has no description.
    """
    skill_md = folder / "SKILL.md"
    if not skill_md.is_file():
        return None

    try:
        content = skill_md.read_text(encoding="utf-8")
    except Exception as e:
        log_warning(f"Cannot read {skill_md}: {e}")
        return None

    fm = _parse_frontmatter(content)

    # Description is required (per Agent Skills spec) — skip without it.
    description = fm.get("description", "").strip()
    if not description:
        log_debug(f"Skipping skill without description: {folder}")
        return None

    name = fm.get("name", folder.name)
    disable = fm.get("disable-model-invocation") is True

    return _SkillMeta(
        name=name,
        description=description,
        skill_file=str(skill_md.resolve()),
        skill_dir=str(folder.resolve()),
        disable_model_invocation=disable,
    )


def _discover_skills(root: Path) -> List[_SkillMeta]:
    """Discover skills under *root*.

    Rules (matching the Agent Skills standard):
    - If root/SKILL.md exists → root is a single skill folder.
    - Otherwise → iterate immediate subdirectories for SKILL.md.
    """
    skills: List[_SkillMeta] = []

    # Single skill folder?
    skill = _load_skill_from_folder(root)
    if skill is not None:
        skills.append(skill)
        return skills

    # Directory of skill folders.
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            skill = _load_skill_from_folder(child)
            if skill is not None:
                skills.append(skill)

    return skills


# ---------------------------------------------------------------------------
# XML escaping
# ---------------------------------------------------------------------------


def _escape_xml(text: str) -> str:
    """Escape special characters for safe inclusion in XML."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


# ---------------------------------------------------------------------------
# SkillToolkit
# ---------------------------------------------------------------------------


class SkillToolkit(Toolkit):
    """Pi-style Skills as a pure-instruction Toolkit (zero tools).

    Scans configured directories for SKILL.md files, extracts ``name`` and
    ``description`` from the YAML frontmatter, and injects an
    ``<available_skills>`` block into the agent's system prompt.  The agent
    then uses its **existing** file and shell tools (e.g.
    ``CodingTools.read_file`` / ``CodingTools.run_shell``) to access skill
    content on demand.

    This is the progressive-disclosure model: only summaries are always in
    context; full instructions are loaded when needed.

    Args:
        dirs: One or more paths to skill folders, or directories that
            contain skill folders.  Each skill folder must have a
            ``SKILL.md`` with YAML frontmatter (``name`` and ``description``
            are required).
    """

    def __init__(
        self,
        dirs: Union[str, Path, Sequence[Union[str, Path]]],
    ):
        # Normalise dirs to a list of Path objects.
        if isinstance(dirs, (str, Path)):
            dir_list: List[Path] = [Path(dirs)]
        else:
            dir_list = [Path(d) for d in dirs]

        # Load skills from all configured directories.
        self._skills: Dict[str, _SkillMeta] = {}
        for dir_path in dir_list:
            resolved = dir_path.resolve()
            if not resolved.exists():
                log_warning(f"SkillToolkit: path does not exist: {resolved}")
                continue
            for skill in _discover_skills(resolved):
                existing = self._skills.get(skill.name)
                if existing is not None:
                    log_warning(
                        f"SkillToolkit: duplicate skill name '{skill.name}', "
                        f"keeping first from {existing.skill_dir}"
                    )
                else:
                    self._skills[skill.name] = skill

        log_debug(f"SkillToolkit: loaded {len(self._skills)} skills")

        # Build instructions string — this is the only thing we contribute.
        instructions = self._build_instructions()

        # Register as a Toolkit with zero tools but with instructions.
        super().__init__(
            name="skills",
            tools=[],
            instructions=instructions,
            add_instructions=True,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_skill(self, name: str) -> Optional[_SkillMeta]:
        """Return metadata for a loaded skill, or None."""
        return self._skills.get(name)

    def get_skill_names(self) -> List[str]:
        """Return the names of all loaded skills."""
        return list(self._skills.keys())

    # ------------------------------------------------------------------
    # Instruction generation
    # ------------------------------------------------------------------

    def _build_instructions(self) -> str:
        """Build the system-prompt instructions block.

        Follows the same XML format as the Pi coding agent so that LLMs
        familiar with that format can immediately understand the convention.
        """
        # Only include skills that are visible to the model.
        visible = [s for s in self._skills.values() if not s.disable_model_invocation]
        if not visible:
            return ""

        lines = [
            "<skills>",
            "The following skills provide specialized instructions for specific tasks.",
            (
                "Use read_file to load the skill's SKILL.md file when "
                "the task matches its description."
            ),
            (
                "When a skill file references a relative path, resolve it "
                "against the skill directory (skill_dir) and use that "
                "absolute path in tool commands."
            ),
            "",
            "<available_skills>",
        ]

        for skill in sorted(visible, key=lambda s: s.name):
            lines.append("  <skill>")
            lines.append(f"    <name>{_escape_xml(skill.name)}</name>")
            lines.append(f"    <description>{_escape_xml(skill.description)}</description>")
            lines.append(f"    <location>{_escape_xml(skill.skill_file)}</location>")
            lines.append(f"    <skill_dir>{_escape_xml(skill.skill_dir)}</skill_dir>")
            lines.append("  </skill>")

        lines.append("</available_skills>")
        lines.append("</skills>")

        return "\n".join(lines)
