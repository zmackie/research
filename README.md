# Research projects carried out by AI tools

Each directory in this repo is a separate research project carried out by an LLM tool - usually [Claude Code](https://www.claude.com/product/claude-code). Every single line of text and code was written by an LLM.

Inspired by [simonw/research](https://github.com/simonw/research) and [Code research projects with async coding agents](https://simonwillison.net/2025/Nov/6/async-code-research/).

I try to include prompts and links to transcripts in [the PRs](../../pulls?q=is%3Apr+is%3Aclosed) that added each report, or in [the commits](../../commits/main/).

*Times shown are in UTC.*

<!--[[[cog
import os
import subprocess
import pathlib
from datetime import datetime, timezone

research_dir = pathlib.Path.cwd()
subdirs_with_dates = []

for d in research_dir.iterdir():
    if d.is_dir() and not d.name.startswith('.'):
        try:
            result = subprocess.run(
                ['git', 'log', '--diff-filter=A', '--follow', '--format=%aI', '--reverse', '--', d.name],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                date_str = result.stdout.strip().split('\n')[0]
                commit_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                subdirs_with_dates.append((d.name, commit_date))
            else:
                subdirs_with_dates.append((d.name, datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc)))
        except Exception:
            subdirs_with_dates.append((d.name, datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc)))

if subdirs_with_dates:
    print(f"## {len(subdirs_with_dates)} research projects\n")

    subdirs_with_dates.sort(key=lambda x: x[1], reverse=True)

    for dirname, commit_date in subdirs_with_dates:
        folder_path = research_dir / dirname
        readme_path = folder_path / "README.md"
        summary_path = folder_path / "_summary.md"

        date_formatted = commit_date.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M')

        github_url = None
        try:
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                origin = result.stdout.strip()
                if origin.startswith('git@github.com:'):
                    origin = origin.replace('git@github.com:', 'https://github.com/')
                if origin.endswith('.git'):
                    origin = origin[:-4]
                github_url = f"{origin}/tree/main/{dirname}"
        except Exception:
            pass

        # Extract title from first H1 header in README, fallback to dirname
        title = dirname
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                for readme_line in f:
                    if readme_line.startswith('# '):
                        title = readme_line[2:].strip()
                        break

        if github_url:
            print(f"### [{title}]({github_url}#readme) ({date_formatted})\n")
        else:
            print(f"### {title} ({date_formatted})\n")

        # Check if summary already exists
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                description = f.read().strip()
                if description:
                    print(description)
                else:
                    print("*No description available.*")
        elif readme_path.exists():
            with open(readme_path, 'r') as f:
                content = f.read().strip()
                # Use first paragraph after title as fallback
                lines = content.split('\n')
                desc_lines = []
                past_title = False
                for line in lines:
                    if line.startswith('# ') and not past_title:
                        past_title = True
                        continue
                    if past_title and line.strip():
                        desc_lines.append(line.strip())
                    if len(desc_lines) >= 3:
                        break
                if desc_lines:
                    print(' '.join(desc_lines))
                else:
                    print("*No description available.*")
        else:
            print("*No description available.*")

        print()
else:
    print("*No research projects yet. Open an issue to start one!*\n")
]]]-->
## 1 research projects

### [JoeBOT: Reverse-Engineering a Neural Network Game Bot from 2000](https://github.com/zmackie/research/tree/main/joebot-investigation#readme) (2026-03-11 12:26)

An investigation into **JoeBOT**, one of the earliest game bots to use artificial neural networks, created for Counter-Strike by Johannes Lampel ([@$3.1415rin](http://joebot.bots-united.com/)) between 2000 and 2005. ## What is JoeBOT? JoeBOT is an AI bot for the original Counter-Strike (a Half-Life mod). While most bots of that era relied entirely on scripted rules and waypoint navigation, JoeBOT was notable for incorporating **feedforward neural networks trained with backpropagation** into its combat and collision avoidance systems. The source code is [available on GitHub](https://github.com/Bots-United/joebot) under the GPL v2 license.

<!--[[[end]]]-->
