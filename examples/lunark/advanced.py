"""P5: ACP advanced features — multi-turn, fork, cancel, list.

Uses raw JSON-RPC over a single hermes acp subprocess so we can exercise
session lifecycle methods that the simple HermesACPClient doesn't cover.

All scenarios run against Qwen3-32B (the most reliable model from prior tests).
"""
from __future__ import annotations
import json, os, subprocess, sys, threading, time, select


class ACPSession:
    """Raw JSON-RPC client over hermes acp stdio."""

    def __init__(self, env: dict | None = None):
        self._proc = subprocess.Popen(
            ["/home/kjs/.local/bin/hermes", "acp"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, env=env or os.environ.copy(),
        )
        self._next_id = 0
        self._lock = threading.Lock()
        self._pending_updates: list[dict] = []
        self._initialize()

    def _send_raw(self, method: str, params: dict | None = None, request_id: int | None = None) -> int:
        with self._lock:
            if request_id is None:
                self._next_id += 1
                request_id = self._next_id
            msg = {"jsonrpc": "2.0", "id": request_id, "method": method}
            if params is not None:
                msg["params"] = params
            self._proc.stdin.write((json.dumps(msg) + "\n").encode())
            self._proc.stdin.flush()
            return request_id

    def _send_notification(self, method: str, params: dict) -> None:
        with self._lock:
            msg = {"jsonrpc": "2.0", "method": method, "params": params}
            self._proc.stdin.write((json.dumps(msg) + "\n").encode())
            self._proc.stdin.flush()

    def _read_line(self, timeout: float) -> dict | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            ready, _, _ = select.select([self._proc.stdout], [], [], 0.2)
            if not ready:
                continue
            line = self._proc.stdout.readline()
            if not line:
                return None
            try:
                return json.loads(line.decode())
            except json.JSONDecodeError:
                continue
        return None

    def _wait_response(self, request_id: int, timeout: float = 60.0) -> dict:
        """Block until matching response. Notifications are buffered."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._read_line(min(2.0, deadline - time.time()))
            if msg is None:
                continue
            if msg.get("id") == request_id:
                if "error" in msg:
                    raise RuntimeError(f"ACP error for {request_id}: {msg['error']}")
                return msg.get("result", {})
            # Buffer notifications (e.g. session/update)
            self._pending_updates.append(msg)
        raise TimeoutError(f"No response for request {request_id} after {timeout}s")

    def _drain_pending(self) -> list[dict]:
        out = self._pending_updates[:]
        self._pending_updates.clear()
        return out

    # ---- handshake ---------------------------------------------------------

    def _initialize(self) -> None:
        rid = self._send_raw("initialize", {
            "protocolVersion": 1,
            "clientCapabilities": {
                "fs": {"readTextFile": False, "writeTextFile": False},
                "terminal": False,
            },
        })
        self._wait_response(rid, 30)

    # ---- session ops -------------------------------------------------------

    def new_session(self, cwd: str = "/tmp") -> str:
        rid = self._send_raw("session/new", {"cwd": cwd, "mcpServers": []})
        result = self._wait_response(rid, 60)
        return result["sessionId"]

    def list_sessions(self) -> list[dict]:
        rid = self._send_raw("session/list", {})
        result = self._wait_response(rid, 30)
        return result.get("sessions", [])

    def fork_session(self, session_id: str, cwd: str = "/tmp") -> str:
        rid = self._send_raw("session/fork", {"sessionId": session_id, "cwd": cwd})
        result = self._wait_response(rid, 60)
        return result["sessionId"]

    def cancel_session(self, session_id: str) -> None:
        # session/cancel is a notification (no response expected)
        self._send_notification("session/cancel", {"sessionId": session_id})

    def prompt(self, session_id: str, text: str, timeout: float = 180.0) -> dict:
        """Send a prompt and collect updates until end_turn or stop."""
        rid = self._send_raw("session/prompt", {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": text}],
        })
        message_chunks: list[str] = []
        thought_chunks: list[str] = []
        events: list[str] = []
        deadline = time.time() + timeout

        # Process buffered notifications first
        for msg in self._drain_pending():
            if msg.get("method") == "session/update":
                upd = msg["params"]["update"]
                kind = upd.get("sessionUpdate", "")
                events.append(kind)
                if kind == "agent_message_chunk":
                    message_chunks.append(upd.get("content", {}).get("text", ""))
                elif kind == "agent_thought_chunk":
                    thought_chunks.append(upd.get("content", {}).get("text", ""))

        while time.time() < deadline:
            msg = self._read_line(min(2.0, deadline - time.time()))
            if msg is None:
                continue
            if msg.get("method") == "session/update":
                upd = msg["params"]["update"]
                kind = upd.get("sessionUpdate", "")
                events.append(kind)
                if kind == "agent_message_chunk":
                    message_chunks.append(upd.get("content", {}).get("text", ""))
                elif kind == "agent_thought_chunk":
                    thought_chunks.append(upd.get("content", {}).get("text", ""))
            elif msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(f"prompt error: {msg['error']}")
                return {
                    "stop_reason": msg.get("result", {}).get("stopReason", ""),
                    "message": "".join(message_chunks),
                    "thought": "".join(thought_chunks),
                    "events": events,
                }
        raise TimeoutError("prompt timed out")

    def close(self) -> None:
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()


# === Test scenarios ============================================================

env = os.environ.copy()
env["HERMES_HOME"] = "/tmp/hermes_Qwen3-32B"
env["LUNARK_API_KEY"] = "vllm-local"

results = []

def record(name: str, passed: bool, detail: str = ""):
    results.append({"test": name, "passed": passed, "detail": detail})
    print(f"  {'✅' if passed else '❌'}  {name}: {detail}", file=sys.stderr)


# ---- Test 1: Multi-turn coherence ------------------------------------------
print("\n=== TEST 1: Multi-turn coherence ===", file=sys.stderr)
try:
    s = ACPSession(env=env)
    sid = s.new_session()
    print(f"  session: {sid}", file=sys.stderr)

    r1 = s.prompt(sid, "Remember the number 42. Just reply 'ok'.", timeout=120)
    print(f"  turn 1 reply: {r1['message'][:80]!r}", file=sys.stderr)

    r2 = s.prompt(sid, "What number did I ask you to remember? Just the number.", timeout=120)
    print(f"  turn 2 reply: {r2['message'][:80]!r}", file=sys.stderr)

    remembered = "42" in r2["message"]
    record("multi-turn coherence", remembered, f"recalled 42: {remembered}")
    s.close()
except Exception as e:
    record("multi-turn coherence", False, f"error: {type(e).__name__}: {str(e)[:100]}")


# ---- Test 2: Session list ---------------------------------------------------
print("\n=== TEST 2: Session list ===", file=sys.stderr)
try:
    s = ACPSession(env=env)
    sid_a = s.new_session()
    sid_b = s.new_session()
    sid_c = s.new_session()
    sessions = s.list_sessions()
    found = {sid_a, sid_b, sid_c} <= {sess.get("sessionId") or sess.get("id") for sess in sessions}
    record("session/list shows new sessions", found, f"created 3, listed {len(sessions)}, all found={found}")
    s.close()
except Exception as e:
    record("session/list shows new sessions", False, f"error: {type(e).__name__}: {str(e)[:120]}")


# ---- Test 3: Session fork divergence ----------------------------------------
print("\n=== TEST 3: Session fork divergence ===", file=sys.stderr)
try:
    s = ACPSession(env=env)
    sid = s.new_session()
    # Build context in original
    s.prompt(sid, "Remember: my favorite color is blue. Just say 'ok'.", timeout=120)

    # Fork it
    forked_sid = s.fork_session(sid)
    print(f"  original: {sid}", file=sys.stderr)
    print(f"  forked:   {forked_sid}", file=sys.stderr)

    # Both should remember "blue"
    r_orig = s.prompt(sid, "What is my favorite color? Just the word.", timeout=120)
    r_fork = s.prompt(forked_sid, "What is my favorite color? Just the word.", timeout=120)

    orig_remembers = "blue" in r_orig["message"].lower()
    fork_remembers = "blue" in r_fork["message"].lower()
    record("fork preserves history", orig_remembers and fork_remembers,
           f"orig={orig_remembers} fork={fork_remembers}")

    # Now diverge: tell fork that color is red
    s.prompt(forked_sid, "Actually my favorite color is now red. Just say 'ok'.", timeout=120)

    # Original should still say blue, fork should say red
    r_orig2 = s.prompt(sid, "What is my favorite color? Just the word.", timeout=120)
    r_fork2 = s.prompt(forked_sid, "What is my favorite color? Just the word.", timeout=120)

    orig_blue = "blue" in r_orig2["message"].lower()
    fork_red = "red" in r_fork2["message"].lower()
    record("fork diverges independently", orig_blue and fork_red,
           f"orig still blue={orig_blue}, fork now red={fork_red}")

    s.close()
except Exception as e:
    record("fork", False, f"error: {type(e).__name__}: {str(e)[:120]}")


# ---- Test 4: Session cancel mid-stream --------------------------------------
print("\n=== TEST 4: Session cancel ===", file=sys.stderr)
try:
    s = ACPSession(env=env)
    sid = s.new_session()
    print(f"  session: {sid}", file=sys.stderr)

    # Start a long prompt and cancel after a brief moment
    long_prompt = "Write a 5000-word essay about the history of computing. Be very detailed."

    # Send the prompt request but don't wait for completion
    rid = s._send_raw("session/prompt", {
        "sessionId": sid,
        "prompt": [{"type": "text", "text": long_prompt}],
    })
    # Let it stream for ~3 seconds then cancel
    time.sleep(3)
    t_cancel = time.time()
    s.cancel_session(sid)
    print(f"  sent cancel at t+{time.time()-t_cancel:.1f}s", file=sys.stderr)

    # Wait for the prompt response (should come quickly with cancelled stop reason)
    try:
        result = s._wait_response(rid, timeout=30)
        stop = result.get("stopReason", "")
        cancelled = stop in ("cancelled", "canceled", "interrupted") or "cancel" in stop.lower()
        record("session/cancel stops prompt", cancelled, f"stopReason={stop!r}")
    except TimeoutError:
        record("session/cancel stops prompt", False, "timed out waiting for cancelled response")
    s.close()
except Exception as e:
    record("session/cancel", False, f"error: {type(e).__name__}: {str(e)[:120]}")


# === Summary ==================================================================
print("\n=== SUMMARY ===", file=sys.stderr)
n = len(results)
p = sum(1 for r in results if r["passed"])
print(f"  {p}/{n} passed", file=sys.stderr)
for r in results:
    print(f"    [{'PASS' if r['passed'] else 'FAIL'}] {r['test']}  {r['detail']}", file=sys.stderr)

with open("/tmp/acp_advanced_results.json", "w") as f:
    json.dump({"summary": {"pass": p, "total": n}, "results": results}, f, indent=2)
