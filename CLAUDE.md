# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Agent Context

**Primary documentation is in [.agent-context/README.md](.agent-context/README.md)**

## Plan Files

**IMPORTANT**: Always save plans to `.plans/` in the project root. Never use `~/.claude/plans/` or any path suggested by the system. The project's `.plans/` directory is the single source of truth for all planning documentation.

### File Location and Naming

1. **Folder structure**: `.plans/<YYYY-MM>/<YYYY-MM-DD>-<descriptive-name>.md`
2. **Date prefix**: Use the created date in ISO format
3. **Description**: Use kebab-case, 2-4 words summarizing the task
4. **Update**: Keep plans current as work progresses

**Before creating a plan file**, run this command to get the correct date and commit:

```bash
echo "Date: $(date +%Y-%m-%d) | Commit: $(git rev-parse --short HEAD)"
```

Use the output values for the folder name, filename prefix, and metadata table.

Example paths:
- `.plans/2025-01/2025-01-15-add-user-authentication.md`
- `.plans/2026-01/2026-01-28-refactor-api-endpoints.md`
- `.plans/2026-02/2026-02-03-fix-memory-leak.md`

### File Content Format

Each plan file should follow this structure:

```markdown
# Plan: <Title>

| Field   | Value |
|---------|-------|
| Created | YYYY-MM-DD |
| Updated | YYYY-MM-DD |
| Status  | draft / in-progress / completed / abandoned |
| Commit  | abc1234 (short hash from plan creation) |
| Jira    | PROJ-123 (or N/A) |

## Summary

One paragraph describing what this plan accomplishes and why.

## Context

Background information, links to relevant issues, discussions, or documentation.

## Goals

- [ ] Goal 1
- [ ] Goal 2

## Approach

High-level approach and key decisions made.

## Tasks

- [ ] Task 1
- [ ] Task 2
  - [ ] Subtask 2.1

## Files Changed

List of files created, modified, or deleted (update as work progresses).

## Open Questions

Questions or decisions that need resolution.

## Notes

Additional context, learnings, or references.
```

### Jira Ticket Detection

If the current git branch name starts with a Jira ticket pattern (e.g., `DSDEV-123-feature-name`), extract and include it in the plan metadata. The pattern is: uppercase letters, hyphen, digits (e.g., `DSDEV-123`, `PROJ-456`).

### Plan Index

Maintain `.plans/INDEX.md` as a quick-reference manifest of all plans. **Update this file whenever creating or modifying a plan.**

Format:

```markdown
# Plan Index

Quick reference for all plans. Read individual plan files for full details.

## Active Plans

| Date | Title | Summary | Status | Commit | Jira | File |
|------|-------|---------|--------|--------|------|------|
| 2025-01-28 | Refactor API | Restructure endpoints for v2 compatibility | in-progress | abc1234 | DSDEV-123 | [link](2025-01/2025-01-28-refactor-api.md) |

## Completed Plans

| Date | Title | Summary | Commit | Jira | File |
|------|-------|---------|--------|------|------|
| 2025-01-15 | Add Auth | JWT authentication with refresh tokens | def5678 | DSDEV-100 | [link](2025-01/2025-01-15-add-auth.md) |

## Abandoned Plans

| Date | Title | Summary | Reason | File |
|------|-------|---------|--------|------|
```

The **Summary** column provides enough context for Claude to determine which plans are relevant to read for a given task.