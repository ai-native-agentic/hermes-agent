"""
Minimal Hermes ACP client — embeds `hermes acp` as a subprocess and talks
JSON-RPC over stdio. Useful as an SDK alternative when you don't want to
import Hermes internals directly.

Usage:
    python examples/acp_client.py "What is 17*23?"

Requires `hermes` on PATH with the [acp] extra installed.
"""
from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class PromptResult:
    message: str = ""
    thought: str = ""
    stop_reason: str = ""
    events: list = field(default_factory=list)


class HermesACPClient:
    def __init__(self, hermes_bin: str = "hermes", cwd: str = ".", env: Optional[dict] = None):
        self._proc = subprocess.Popen(
            [hermes_bin, "acp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env or os.environ.copy(),
        )
        self._next_id = 0
        self.session_id: Optional[str] = None
        self.cwd = cwd
        self._initialize()
        self._new_session(cwd)

    # -- low-level JSON-RPC --------------------------------------------------

    def _send(self, method: str, params: dict, request_id: Optional[int] = None) -> int:
        if request_id is None:
            self._next_id += 1
            request_id = self._next_id
        msg = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        self._proc.stdin.write((json.dumps(msg) + "\n").encode())
        self._proc.stdin.flush()
        return request_id

    def _recv(self, timeout: float = 30.0) -> Optional[dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            ready, _, _ = select.select([self._proc.stdout], [], [], 0.3)
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

    def _recv_response(self, request_id: int, timeout: float = 30.0) -> dict:
        """Block until a response with matching id arrives."""
        while True:
            msg = self._recv(timeout)
            if msg is None:
                raise TimeoutError(f"No response for request {request_id}")
            if msg.get("id") == request_id:
                if "error" in msg:
                    raise RuntimeError(f"ACP error: {msg['error']}")
                return msg.get("result", {})

    # -- handshake -----------------------------------------------------------

    def _initialize(self) -> None:
        rid = self._send(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
            },
        )
        self._recv_response(rid)

    def _new_session(self, cwd: str) -> None:
        rid = self._send("session/new", {"cwd": cwd, "mcpServers": []})
        result = self._recv_response(rid, timeout=60)
        self.session_id = result.get("sessionId")
        if not self.session_id:
            raise RuntimeError(f"session/new returned no sessionId: {result}")

    # -- prompt --------------------------------------------------------------

    def prompt(self, text: str, timeout: float = 300.0) -> PromptResult:
        """Send a prompt and collect streamed updates until end_turn."""
        rid = self._send(
            "session/prompt",
            {
                "sessionId": self.session_id,
                "prompt": [{"type": "text", "text": text}],
            },
        )
        result = PromptResult()
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._recv(timeout)
            if msg is None:
                break
            if msg.get("method") == "session/update":
                upd = msg["params"]["update"]
                kind = upd.get("sessionUpdate", "")
                result.events.append(kind)
                if kind == "agent_message_chunk":
                    result.message += upd.get("content", {}).get("text", "")
                elif kind == "agent_thought_chunk":
                    result.thought += upd.get("content", {}).get("text", "")
            elif msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(f"ACP error: {msg['error']}")
                result.stop_reason = msg.get("result", {}).get("stopReason", "")
                return result
        return result

    def stream(self, text: str, timeout: float = 300.0) -> Iterator[tuple[str, str]]:
        """Yield (kind, text) tuples as updates arrive. Kind in {message, thought, event}."""
        rid = self._send(
            "session/prompt",
            {
                "sessionId": self.session_id,
                "prompt": [{"type": "text", "text": text}],
            },
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self._recv(timeout)
            if msg is None:
                break
            if msg.get("method") == "session/update":
                upd = msg["params"]["update"]
                kind = upd.get("sessionUpdate", "")
                if kind == "agent_message_chunk":
                    yield "message", upd.get("content", {}).get("text", "")
                elif kind == "agent_thought_chunk":
                    yield "thought", upd.get("content", {}).get("text", "")
                else:
                    yield "event", kind
            elif msg.get("id") == rid:
                if "error" in msg:
                    raise RuntimeError(f"ACP error: {msg['error']}")
                yield "done", msg.get("result", {}).get("stopReason", "")
                return

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _cli() -> None:
    text = " ".join(sys.argv[1:]) or "What is 17*23? Reply with just the number."
    with HermesACPClient(cwd=os.getcwd()) as client:
        print(f"[session: {client.session_id}]\n", file=sys.stderr)
        for kind, payload in client.stream(text):
            if kind == "message":
                print(payload, end="", flush=True)
            elif kind == "thought":
                print(f"\033[2m{payload}\033[0m", end="", file=sys.stderr, flush=True)
            elif kind == "event":
                print(f"\n[{payload}]", file=sys.stderr)
            elif kind == "done":
                print(f"\n\n[done: {payload}]", file=sys.stderr)
        print()


if __name__ == "__main__":
    _cli()
