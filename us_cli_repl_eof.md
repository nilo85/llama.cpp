# us_cli_repl_eof — llama-cli REPL infinite loop on EOF stdin (upstream bug)

**Type:** Bug fix (upstream candidate); unrelated to M14 expert split, tracked separately.
**Priority (2026-10-02):** P2, OPEN, deferred until M14 lands.
**Status:** Root cause fully traced in our tree. Workaround in place for all local tests (pipe `/exit` on stdin).

## Persona
Me (and any CI/automation): running `llama-cli -m model -p 'prompt'` with non-TTY stdin (podman/docker `run` without `-t`, piped stdin, cron jobs) for one-shot completions or parity tests.

## Story
As an automated llama.cpp user,
I want `llama-cli -p 'prompt'` to process the prompt, print the generation, and exit,
so my pipelines terminate instead of hanging forever and flooding stdout with gigabytes of prompt garbage.

## Symptom
- `llama-cli ... -p '...'` with stdin = /dev/null or a closed pipe: after the one-shot generation completes, the process never exits. It spins at 99% of one core printing the REPL prompt `> \x1b[0m` to stdout ~1.6M times/sec (~11 MB/s). A 14-minute hang produced a 9.5 GB stdout file.
- Killed only by external `timeout`; exit code 124.
- Parity diffs of the stdout files still "pass" because both sides hang identically and the generated text (written before the hang) matches - the bug is silent unless file size or exit code is checked.

## Root cause (traced 2026-10-02, branch us-otgen-expert-ot)
1. `llama_cli` (tools/cli/cli.cpp:67) always calls `ctx_cli.run()`; there is no one-shot branch.
2. `cli_context::run()` (tools/cli/cli-context.cpp:380) unconditionally enters the interactive `while (true)` loop; the `-p` prompt is consumed as the first turn then cleared (`params.prompt.clear(); // only use it once`, cli-context.cpp:~486).
3. Subsequent turns call `ui::user_turn::read_input()` (tools/cli/cli-ui.h:166), which prints the `> ` prompt unconditionally, then `console::readline()`.
4. `console::readline_advanced` (common/console.cpp:754, EOF case at ~line 813: `input_char == WEOF -> end_of_stream = true`) returns `false` at EOF, so `read_input` returns an empty string.
5. The REPL loop treats that as an empty user message: `if (buffer.empty()) { continue; }` (cli-context.cpp:~507) - it never distinguishes "empty line" from "EOF", so it loops forever, re-printing `> ` each iteration.

## Workaround (used by all local M14 test runs)
Pipe the exit command on stdin so the REPL gets a real line and breaks cleanly:
```sh
echo /exit | timeout 900 llama-cli ... -p 'prompt'
```
After the first turn the loop reads `/exit`, matches `string_starts_with(buffer, "/exit")`, and exits.

## Proposed fix
- Propagate EOF out of `read_input` (e.g. `std::optional<std::string>` or a bool out-param) and `break` the REPL loop on EOF instead of `continue`.
- Alternatively/additionally: when stdin is not a TTY, exit after the predefined `-p` turn (classic one-shot semantics) - this restores the behavior older llama-completion-style runs had.
- Before submitting upstream: search llama.cpp issues/PRs for existing reports (llama-cli interactive EOF / infinite loop / non-tty); check current master - this fork tracks a recent master, so the bug may already be fixed or discussed upstream.

## Acceptance criteria
1. `echo /exit | llama-cli -m M -p 'x'` exits with code 0 after the generation (already works via workaround).
2. `llama-cli -m M -p 'x' < /dev/null` exits with code 0 after the generation (no hang, no `> ` flood).
3. Interactive TTY use unchanged: REPL still works, empty lines still skipped, `/exit` still works.
4. Upstream issue filed (or existing one found and referenced) with the root-cause trace above.

## Definition of done
- Fix implemented locally (small, in tools/cli/cli-context.cpp + tools/cli/cli-ui.h) and tested per acceptance criteria, OR upstream fix found and cherry-picked.
- If upstream rejects/changes the design, document the outcome here and keep the `/exit` workaround in test scripts.

## TODO
- [x] Create story + root-cause trace
- [x] Workaround applied to M14 test suite (`echo /exit |`)
- [ ] Search upstream issues/PRs for existing report
- [ ] Implement EOF-break fix (or cherry-pick upstream fix)
- [ ] Verify acceptance criteria 1-3
