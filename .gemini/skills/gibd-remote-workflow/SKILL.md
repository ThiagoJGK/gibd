---
name: gibd-remote-workflow
description: Establishes a highly technical, disciplined remote workflow for coding agents working on the GIBD WEB repository. Mandates systematic research, implementation plans, progressive task checklists, and validation walkthroughs.
---

# GIBD Remote Workflow Skill

Technical guidelines, constraints, and operational workflows for AI agents collaborating on the GIBD WEB (MITTCA) repository under the 3-Dev remote setup.

---

## 🎯 Core Premise & Operational Mandate

Every task assigned to an AI agent in this repository must undergo a structured lifecycle of planning, modular execution, strict verification, and documented code review. 

### Branch & Deployment Rules
- **Rama `main` = Producción (Vercel):** The `main` branch is connected directly to Vercel hosting. Any code merged into `main` goes immediately live. The integrity of `main` is absolute. No untested, broken, or experimental code is allowed to reach `main`.
- **Preview Deployments:** Vercel generates preview URLs for every Pull Request. Agents must design and test features keeping this environment in mind.

---

## 📋 The 4-Phase Agent Pipeline

### Phase 1: Research & Planning (NO Modification Phase)
Before editing any source code file, the agent must thoroughly research the task and outline the design.

1. **Research & Scan:** Look through relevant files, check current styling, routing, and backend structure. Do not start coding.
2. **Generate Implementation Plan:** Create an `implementation_plan.md` in the conversation's brain artifacts directory. The plan must follow this exact format:
   - **Goal Description:** Brief description of the problem and target.
   - **User Review Required:** Highlight any breaking changes, performance risks, or architectural decisions.
   - **Open Questions:** Ask the developer any clarifying questions that are critical to the implementation.
   - **Proposed Changes:** Group files by component, marked with `[MODIFY]`, `[NEW]`, or `[DELETE]`. Use precise file links.
   - **Verification Plan:** Outline automated test commands and manual testing steps (such as Vercel preview testing).
3. **Wait for Approval:** Stop execution and wait for the developer to explicitly review and approve the plan.

---

### Phase 2: Progressive Execution & Checklist Tracking
Once the user approves the implementation plan, the agent must track progress visibly and proceed modularly.

1. **Create the Task Tracker:** Create a `task.md` file in the conversation's brain artifacts directory.
2. **Task Format:** Organize tasks using standard checkboxes:
   - `- [ ]` for uncompleted tasks
   - `- [/]` for in-progress tasks
   - `- [x]` for completed tasks
3. **Modular Code Writes:** Edits must be contiguous and minimal where possible. If modifying React or styling components, strict adherence to the `gibd-frontend-expert` design guidelines is required (pill geometries, flat OLED-black themes, hardware-accelerated animations, NO Tailwind utility styling unless explicitly demanded, NO box-shadows).
4. **Validation:** Continuously run and verify changes locally to prevent regressions.

---

### Phase 3: Walkthrough & Pull Request Preparation
A task is not complete until it is fully documented for the human team.

1. **Generate Walkthrough:** Create a `walkthrough.md` in the brain artifacts folder detailing:
   - Exact files modified and created.
   - Testing methodologies executed and proof of success (terminal test logs, validation outcomes).
   - Detailed visual progress if the change affects the frontend interface.
2. **Draft the Pull Request (PR):** Write a concise, professional PR description in Markdown. This description should link to the walkthrough and explain:
   - *Why* the changes were made.
   - *How* the changes were implemented.
   - Instructions for the reviewer to run and test the PR.

---

### Phase 4: AI-Assisted Code Review (Revisor Role)
When a developer instructs their agent to review a peer's Pull Request:

1. **Diff Auditing:** Scan the changes introduced by the target branch.
2. **System Verification:** Ensure the proposed changes:
   - Compile successfully without TypeScript strict errors.
   - Conform to the flat design language and component checklist.
   - Do not introduce CPU-intensive layout reflows (no animating layout properties).
   - Are free of security vulnerabilities or hidden bugs.
3. **Structured Review Report:** The reviewing agent must write a constructive, polite markdown feedback report pointing out any potential edge cases or bugs, and outlining what is approved.
