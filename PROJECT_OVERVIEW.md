# Sonaire - Technical Project Overview

Sonaire is an AI-assisted audio-processing project. The production worker in this repository still uses the historical internal name `StemForge` in code, environment variables, workflows, and deployment contracts. Those identifiers are intentionally unchanged because renaming them casually could break the operating system.

## What I own

My role is product and technical direction rather than conventional line-by-line software development.

I define the desired audio behavior, translate it into concrete requirements for AI coding agents, choose tools and models, review implementations, run tests and renders, diagnose failures, compare outputs, and decide whether a result is acceptable.

Claude Code and other AI development tools perform substantial implementation work. The project is useful evidence of my ability to direct, evaluate, troubleshoot, and iterate on complex AI-assisted software, not a claim that I personally authored every line.

## System overview

The worker combines:

- RunPod Serverless GPU execution
- Python audio and DSP pipelines
- stem inspection and reconstruction checks
- Demucs separation
- WhisperX alignment
- loudness and artifact analysis
- mastering and stem-remix workflows
- optional neural-vocoder processing
- structured JSON jobs and reports
- GitHub-based relay workflows
- automated validation and quality gates

## Engineering practices in the repository

The project uses explicit validation before GPU work is dispatched, structured quality checks after processing, tests around engine behavior, documented operator/engine responsibilities, version and build verification, and conservative fallbacks when a processing path fails.

Recent work has included diagnosing unsafe URL handling, tightening ingest validation, distinguishing queued cloud jobs from slow renders, and adjusting processing after listening tests exposed modulation that objective gates did not catch.

## Why the repository remains private

The repository contains production-oriented operational history, deployment contracts, request/result infrastructure, and implementation details. It should not be made public simply to provide a portfolio link.

A sanitized public technical case study can be created separately if the project is used as a public portfolio artifact.

## Naming

Product name: **Sonaire**

Historical/internal worker name: **StemForge**

The distinction is documentation-only until a deliberate migration is planned and tested.
