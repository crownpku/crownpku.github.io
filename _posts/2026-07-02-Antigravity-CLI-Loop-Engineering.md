---
layout: post
comments: true
title: Antigravity CLI: From Vibe Coding to Autonomous Loop Engineering
published: true
---

**The transition from Gemini CLI to Google’s Antigravity CLI represents a fundamental shift in how developers interact with artificial intelligence, marking the transition from human-driven "Vibe Coding" to fully autonomous "Loop Engineering".** While vibe coding relies on turn-by-turn chat interactions where the developer holds the steering wheel as the manual driver, loop engineering introduces a level of abstraction above the agent harness. It replaces manual prompting with recursive, goal-driven automation. Under this paradigm, the developer defines a high-level purpose (such as optimizing thread performance or resolving visual regressions) and lets a closed-loop system autonomously plan, execute, compile, test, observe outcomes, and self-correct until the goal is strictly verified.

During the **Agentic Architect Sprint 2026**, I put this new meta to the test. Using the Go-based Antigravity CLI (`agy`) and the lightning-fast **Gemini 3.5 Flash** model, I set out to build the **Manulife Tower Building Game**—a high-fidelity, single-player, browser-based simulation of the iconic Grade-A skyscraper at 8 Cross Street in Singapore's CBD.

The entire development process was driven by Antigravity’s **Fully Autonomous Goal Execution (`/goal`)** mode. Instead of micromanaging the AI, I let the agent harness coordinate multiple background subagents to trace execution stutters, design custom test scenarios, analyze TypeScript compiler errors, and dynamically refactor game state structures until the game achieved a locked, silky-smooth 60 FPS.

The clean, fully optimized production version is publicly accessible here:  
👉 **[Play the Manulife Tower Building Game on Google Cloud](https://manulife-tower-building-game.storage.googleapis.com/index.html)**

---

## 1. Unpacking the Loop Engineering Architecture in Antigravity CLI

As articulated by software experience leaders like Addy Osmani, a robust loop-engineered system is comprised of five key primitives operating above a persistent state layer:

1.  **Automations**: Scheduled background execution runners (like `/goal` or `/loop` directives) that drive work asynchronously without human prompting.
2.  **Worktrees**: Isolated working directories (`git worktree`) that allow parallel subagents to write code concurrently without workspace collisions.
3.  **Skills**: Directory-based configurations (`SKILL.md`) that codify project conventions and style constraints, preventing the model from making blind assumptions.
4.  **Connectors & Plugins**: Protocol-level bridges (such as Model Context Protocol or MCP servers) that hook the agent directly into system tools, databases, and APIs.
5.  **Sub-agents**: Dedicated helper instances that separate the "maker" (the agent writing the code) from the "checker" (the agent auditing security or verifying test coverage).
6.  **State Memory**: Persistent, disk-based Markdown tracking files (such as `task.md` or `walkthrough.md`) that preserve context across long-running runs, ensuring the agent never forgets its history.

```
                 +---------------------------------------------+
                 |            High-Level Goal Input            |
                 |      "Optimize build performance to 0ms"    |
                 +----------------------+----------------------+
                                        |
                                        v
                 +----------------------+----------------------+
                 |          Agent Planning Phase                |
                 |    - Static Analysis & Code Inspections     |
                 +----------------------+----------------------+
                                        |
                                        v
                 +----------------------+----------------------+
                 |         Autonomous Workspace (Maker)        |
                 |    - Write Optimized TS Code in Scratch     |
                 +----------------------+----------------------+
                                        |
                                        v
                 +----------------------+----------------------+
                 |       Verification Engine (Checker)         |
                 |    - Run Typechecks & Compiler (Vite)       |
                 |    - Run Unit Test Suites (Vitest)          |
                 +----------------------+----------------------+
                                        |
                     +------------------+------------------+
                     |                                     |
              (Test Fails)                               (100% Pass)
                     v                                     v
         +-----------+-----------+                 +-------+-------+
         |     Self-Correction   |                 | Auto-Merge &  |
         |  - Parse Stack Trace  |                 |  GCP Deploy   |
         |  - Refine Logic       |                 +---------------+
         +-----------+-----------+
                     |
                     +-------------------------------------+
```

In the **Manulife Tower** project, the loop’s heartbeat was driven entirely within the Antigravity CLI harness. I invoked the system with a single directive:

```bash
agy /goal "Analyze all rendering, physics, and state-saving stutters in the game scene. Implement high-performance optimizations and verify they pass the unit test suite."
```

From this single command, the agent initialized a **closed loop of autonomous execution**. It scanned the directories, located the source code, ran diagnostic builds, analyzed test failure logs, and rewrote core simulation components—all while I sat back and watched the status bar update.

---

## 2. Walkthrough: The Manulife Tower Simulation Game

The **Manulife Tower Building Game** is a client-side, 100% serverless and offline-first simulation of a modern CBD skyscraper. Players act as the chief architect of Singapore's real-world Grade-A green office tower at 8 Cross Street. 

### Thematic & Architecture Highlights:
*   **"City in a Garden" Aesthetic**: Glassmorphic panels, deep emerald green accents, and slate borders reflecting the biophilic architectural identity of the real building.
*   **Underground MRT Integration**: A custom subterranean **Telok Ayer MRT Station (DT18)**, complete with fully animated silver and azure-blue Downtown Line trains arriving on schedules to drop off commuters.
*   **Fully-Suited CBD Commuters**: Commuters are styled as Singapore business professionals wearing blazers and neckties, transitioning dynamically through the building.
*   **Premium Glass Elevators**: Moving capsule cabins designed with chrome structural framing, brushed steel interior backings, spotlights, and dynamic crowd silhouettes representing occupancy.
*   **Local Storage Portfolios**: Absolute client-side data persistence using the browser’s `localStorage`, allowing players to save and load their towers completely offline.

---

## 3. Case Studies in Closed-Loop Self-Correction

The true magic of the Antigravity CLI occurred when the system hit complex execution failures. Rather than stopping to ask me for advice, the CLI's checker-maker architecture continuously inspected compile errors, parsed stack trace exceptions, and evaluated simulated game outcomes to refine its own code.

Here are three deep-dive case studies of how the autonomous loop engineered its way out of critical performance and visual stutters:

### Case Study 1: The 7:00 AM Workday Spawning Glitch (Visual Analysis Loop)
During testing, a bizarre visual regression appeared: around 7:00 AM on workdays, a massive crowd of Sims would suddenly "flash" or spawn visible inside their offices and corridors, only to immediately vanish a split-second later.

```
[Simulation Reset (7:00 AM)] ──> [Sims Snap to Office Desks] ──> [Phaser Detects Floor Delta]
                                                                           │
                                                                           v
[Visual "Flash Out" Glitch] <── [Render Filter Fails to Hide] <── [isWalking Set to True]
```

*   **How the Loop Diagnosed It**: The agent reviewed `GameScene.ts` and tracked down the daily simulation tick state changes. It analyzed that at 7:00 AM, the engine executed `resetSimRuntimeState` to reset commuters to `STATE_MORNING_GATE` (parked inside offices) and updated their `selectedFloor`. However, because their horizontal coordinate targets shifted, the visual interpolation engine flagged their floor state change as a transit exit, unconditionally setting `entry.isWalking = true`. This made hundreds of inactive Sims momentarily visible at their desks before they vanished.
*   **The Autonomous Self-Correction**: The agent designed a visibility check in `GameScene.ts` (`drawSims`). It inspected whether the Sim's active state matched any parked or inactive states (such as `STATE_MORNING_GATE` or `STATE_AT_WORK`). If inactive, the agent bypassed the interpolation loop entirely, snapped their coordinates directly, and hardcoded `isWalking = false`.
*   **Outcome Verification**: The compiler ran a headless compile, verified that the new logic resolved the type checking constraints, and verified that the morning commute now transitioned seamlessly with zero ghost flashing.

### Case Study 2: Erase Lag and Deferred Pathfinder Batching (Performance Tuning Loop)
The game was plagued by severe UI freeze-ups when players used the erase tool to quickly drag and clear large blocks of cells. 

*   **How the Loop Diagnosed It**: The CLI's profiling subagent analyzed the execution stack of the eraser tool. It discovered that dragging the eraser fired a rapid series of individual `remove_tile` commands in succession. In the simulation worker, each individual command synchronously called `runGlobalRebuilds()`, which forced expensive, CPU-intensive pathfinding grid reconstructions, transfer group cache calculations, and entity sorting on *every single pixel frame*.
*   **The Autonomous Self-Correction**: The agent realized that running these calculations per-cell was highly redundant during rapid click-drags. It introduced two transient state flags to the simulation's `WorldState`: `deferGlobalRebuilds` and `rebuildsDirty`. It then updated `runGlobalRebuilds` to check if rebuilding was deferred; if so, it flagged the state as dirty and exited immediately. The agent then implemented a **Deterministic Throttled Rebuild** system that delayed global flushes during drag events, batching the calculations to execute exactly once every 10 ticks (approx. 500ms) on a strict, deterministic tick boundary.
*   **Outcome Verification**: The agent ran the local Vitest test suites. When the compiler initially complained about out-of-date reachability tables in the cathedral stress-test, the agent intercepted the error, refined the clean-up routines to trigger `flushGlobalRebuilds()` automatically during state un-deferrals, and successfully ran all 113 unit tests to a green state.

```typescript
// Deterministic Throttled Rebuild Hook designed autonomously by agy
public runGlobalRebuilds(): void {
  if (this.state.world.deferGlobalRebuilds) {
    this.state.world.rebuildsDirty = true;
    return; // Early return prevents thread blocking during rapid drags
  }
  
  this.rebuildRouteReachabilityTables();
  this.rebuildRuntimeSims();
  this.rebuildTransferGroupCache();
  this.state.world.rebuildsDirty = false;
}
```

### Case Study 3: The 10x Speed Timer Throttling (Environment Feedback Loop)
When accelerating the simulation speed to `3x` or `10x`, the visual animations sped up perfectly, but the in-game world clock and day/night cycle progressed at a sluggish, erratic pace.

*   **How the Loop Diagnosed It**: The agent investigated the timing loop inside `lockstepSession.ts`. It discovered that standard browsers strictly clamp the minimum execution interval of `setTimeout`/`setInterval` to **4ms**. At `10x` speed, the calculated interval between simulation ticks was less than 5ms, which triggered aggressive browser timer throttling and caused actual tick deliveries to drop to roughly `3x` or `4x` in practice. Furthermore, the local Cloudflare Worker backend (`TowerRoom.ts`) was experiencing the exact same event-loop dispatching overhead.
*   **The Autonomous Self-Correction**: Rather than trying to force faster timer intervals, the agent designed a robust **multi-tick timing resolution**. Under `10x` speed, it configured the loop to run a stable, browser-friendly 15ms interval but advanced **multiple logical simulation ticks (3 ticks) within a single loop callback**. It then applied this exact same multi-tick architecture to the local backend server in `TowerRoom.ts`.
*   **Outcome Verification**: The agent built both the client and the worker server locally, verified that their states stayed in perfect lockstep, and confirmed that the top-right world clock now advanced at a perfectly synchronized, proportional rate.

---

## 4. Continuous Verification: Passing the Ultimate Test

A core principle of loop engineering is that **"done" is a verified claim, not a blind assertion.** 

To guarantee that these deep performance optimizations didn't introduce game state desynchronization or break the lockstep engine, the Antigravity CLI harness continuously verified its work against our comprehensive local test suite:

```
========================= VITEST TEST RESULTS =========================
✓ apps/worker/src/sim/trace.test.ts (11 test files passed)
✓ 113 / 113 tests passed (100% green)

Performance Milestone:
- Skyscraper initialization (10,400 active tiles across 104 floors)
- Setup execution time: 225ms (dropped from 4,440ms - 20x setup speedup!)
=======================================================================
```

By deferring global pathfinder sweeps across massive grid placements and only performing a unified flush before spawning active elevator cabs, the agent reduced the initialization setup time in the unit tests from **4,440ms down to a mere 225ms (a stunning 20x speedup)**.

---

## 5. Conclusion: Designing Systems, Not Writing Code

My journey building the **Manulife Tower Building Game** proved that the loop-engineered capabilities of the Antigravity CLI represent a massive leap forward for developer leverage. By using the `/goal` mode, I stopped acting as an active coder and transitioned into a **systems and feedback architect.** 

I defined the parameters, provided the environment, and established the quality gates, while the autonomous loops compiled, tested, analyzed errors, and self-corrected their way to a flawless production build. 

The era of babysitting AI coding assistants is over. It is time to start building loops.

🔗 **[Start Construction on Your Manulife Tower Now!](https://manulife-tower-building-game.storage.googleapis.com/index.html)**
