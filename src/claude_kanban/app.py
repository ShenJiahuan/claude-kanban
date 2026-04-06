#!/usr/bin/env python3
"""Claude Code Agent Kanban - Web dashboard for monitoring Claude Code sessions across servers."""

import glob
import json
import os
import platform
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import paramiko
import yaml
from flask import Flask, jsonify, request, send_from_directory

_PKG_DIR = Path(__file__).resolve().parent
_DATA_DIR = Path(os.environ.get("KANBAN_DATA_DIR", str(Path.home() / ".claude-kanban")))

app = Flask(__name__, static_folder=str(_PKG_DIR / "static"))

CONFIG_PATH = os.environ.get("KANBAN_CONFIG", str(_DATA_DIR / "config.yaml"))
CACHE_TTL = 30  # seconds
_cache = {"data": None, "ts": 0, "lock": threading.Lock()}

# Summary cache: {sessionId: {"messageCount": N, "summary": {...}}}
# Persisted to disk so summaries survive restarts.
SUMMARY_CACHE_PATH = _DATA_DIR / ".summary_cache.json"
_summary_cache = {"data": {}, "lock": threading.Lock()}


def _load_summary_cache():
    if SUMMARY_CACHE_PATH.exists():
        try:
            with open(SUMMARY_CACHE_PATH) as f:
                _summary_cache["data"] = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass


def _save_summary_cache():
    try:
        with open(SUMMARY_CACHE_PATH, "w") as f:
            json.dump(_summary_cache["data"], f, ensure_ascii=False)
    except OSError as e:
        print(f"[WARN] Failed to save summary cache: {e}")

# Track sessions seen as running — when they disappear from active, move to completed
_known_sessions = {"lock": threading.Lock(), "running": {}, "completed": {}}


def _default_config():
    return {"include_local": True, "servers": []}


def load_config():
    path = Path(CONFIG_PATH)
    if not path.exists():
        return _default_config()
    with open(path) as f:
        return yaml.safe_load(f) or _default_config()


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Local collection
# ---------------------------------------------------------------------------

def collect_local():
    """Collect currently active Claude Code sessions from the local machine.

    Only returns sessions that have an entry in ~/.claude/sessions/ (i.e. running).
    Completed sessions are tracked separately via _known_sessions.
    """
    home = Path.home()
    sessions_dir = home / ".claude" / "sessions"
    projects_dir = home / ".claude" / "projects"
    server_name = platform.node() or "localhost"

    if not sessions_dir.exists():
        return []

    sessions = []
    for sf in sessions_dir.glob("*.json"):
        try:
            meta = json.loads(sf.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        session_id = meta.get("sessionId", "")
        if not session_id:
            continue

        # Skip subagent sessions
        if session_id.startswith("agent-"):
            continue

        pid = meta.get("pid")
        alive = _pid_alive_local(pid) if pid else False
        cwd = meta.get("cwd", "")
        started_at = meta.get("startedAt", 0)

        # Find and parse matching JSONL
        jsonl_path = _find_jsonl(session_id, projects_dir)
        task_summary = ""
        last_activity = started_at
        message_count = 0
        excerpt = []
        token_usage = {"totalInputTokens": 0, "totalOutputTokens": 0, "cacheCreationTokens": 0, "cacheReadTokens": 0}
        if jsonl_path:
            task_summary, last_activity, message_count, excerpt, token_usage = _parse_jsonl(jsonl_path, started_at)

        # Skip summarization sessions
        if task_summary and "concise status reporter" in task_summary:
            continue

        started_dt = datetime.fromtimestamp(started_at / 1000, tz=timezone.utc).isoformat() if started_at else ""
        last_dt = ""
        if last_activity and last_activity != started_at:
            last_dt = datetime.fromtimestamp(last_activity / 1000, tz=timezone.utc).isoformat()

        sessions.append({
            "sessionId": session_id,
            "pid": pid,
            "cwd": cwd,
            "project": os.path.basename(cwd) if cwd else "",
            "startedAt": started_dt,
            "lastActivity": last_dt,
            "kind": meta.get("kind", ""),
            "entrypoint": meta.get("entrypoint", ""),
            "alive": alive,
            "taskSummary": task_summary,
            "messageCount": message_count,
            "tokenUsage": token_usage,
            "conversationExcerpt": excerpt,
            "server": server_name,
        })

    return sessions


def _find_jsonl(session_id, projects_dir):
    """Find the JSONL conversation file for a session."""
    if not projects_dir or not projects_dir.exists():
        return None
    for jsonl in projects_dir.rglob(f"{session_id}.jsonl"):
        return jsonl
    return None




def _pid_alive_local(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False



def _extract_text_content(content):
    """Extract plain text from a message content field."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text":
                    texts.append(c.get("text", ""))
                elif c.get("type") == "tool_result":
                    # Skip tool results to save tokens
                    pass
        return " ".join(texts).strip()
    return ""


def _parse_jsonl(path, started_at):
    """Parse a JSONL file to extract messages, activity info, and conversation excerpts."""
    task_summary = ""
    last_ts = started_at
    message_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    cache_creation_tokens = 0
    cache_read_tokens = 0
    # Collect conversation messages for AI summarization
    first_messages = []  # first few user messages (the task)
    recent_messages = []  # rolling window of recent messages

    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = entry.get("type")
                role = entry.get("message", {}).get("role", msg_type)

                # Count user/assistant messages
                if role in ("user", "assistant"):
                    message_count += 1

                # Track token usage from assistant messages
                if role == "assistant":
                    usage = entry.get("message", {}).get("usage", {})
                    total_input_tokens += usage.get("input_tokens", 0)
                    total_output_tokens += usage.get("output_tokens", 0)
                    cache_creation_tokens += usage.get("cache_creation_input_tokens", 0)
                    cache_read_tokens += usage.get("cache_read_input_tokens", 0)

                # Extract text content for conversation excerpt
                if role in ("user", "assistant"):
                    text = _extract_text_content(entry.get("message", {}).get("content", ""))
                    if text and not text.startswith("<"):
                        msg_entry = {"role": role, "text": text[:500]}
                        # Keep first 3 messages (task context)
                        if len(first_messages) < 3:
                            first_messages.append(msg_entry)
                        # Keep last 6 messages (recent progress)
                        recent_messages.append(msg_entry)
                        if len(recent_messages) > 6:
                            recent_messages.pop(0)

                # Extract first user message as raw task summary fallback
                if not task_summary and msg_type == "user":
                    text = _extract_text_content(entry.get("message", {}).get("content", ""))
                    if text and not text.startswith("<"):
                        task_summary = text[:300]

                # Track last activity via timestamps in snapshot entries
                snapshot = entry.get("snapshot", {})
                if snapshot:
                    ts_str = snapshot.get("timestamp", "")
                    if ts_str:
                        try:
                            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            ts_ms = int(dt.timestamp() * 1000)
                            if ts_ms > last_ts:
                                last_ts = ts_ms
                        except (ValueError, OSError):
                            pass
    except OSError:
        pass

    # Also check file modification time as a proxy for last activity
    try:
        mtime_ms = int(path.stat().st_mtime * 1000)
        if mtime_ms > last_ts:
            last_ts = mtime_ms
    except OSError:
        pass

    # Build conversation excerpt: first messages + ... + recent messages (deduplicated)
    excerpt = list(first_messages)
    first_texts = {m["text"] for m in first_messages}
    for m in recent_messages:
        if m["text"] not in first_texts:
            excerpt.append(m)

    token_usage = {
        "totalInputTokens": total_input_tokens,
        "totalOutputTokens": total_output_tokens,
        "cacheCreationTokens": cache_creation_tokens,
        "cacheReadTokens": cache_read_tokens,
    }
    return task_summary, last_ts, message_count, excerpt, token_usage


# ---------------------------------------------------------------------------
# Remote collection via SSH
# ---------------------------------------------------------------------------

def collect_remote(server_conf):
    """Collect Claude Code sessions from a remote server via SSH."""
    host = server_conf["host"]
    port = server_conf.get("port", 22)
    user = server_conf.get("user", os.getenv("USER", "root"))
    key_path = os.path.expanduser(server_conf.get("key", "~/.ssh/id_rsa"))
    label = server_conf.get("label", host)

    script = _remote_script()

    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs = {
            "hostname": host,
            "port": port,
            "username": user,
            "timeout": 10,
            "banner_timeout": 10,
        }
        if os.path.exists(key_path):
            connect_kwargs["key_filename"] = key_path
        else:
            # Fall back to SSH agent
            connect_kwargs["allow_agent"] = True

        client.connect(**connect_kwargs)
        _, stdout, stderr = client.exec_command(f"python3 -c {_shell_quote(script)}", timeout=30)
        output = stdout.read().decode()
        client.close()

        if not output.strip():
            return []

        sessions = json.loads(output)
        for s in sessions:
            s["server"] = label
        return sessions

    except Exception as e:
        print(f"[WARN] Failed to collect from {label} ({host}): {e}")
        return [{"server": label, "error": str(e)}]


def _shell_quote(s):
    """Shell-quote a string for remote execution."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _remote_script():
    """Python script to run on remote servers to collect only active session data."""
    return r'''
import json, os, sys
from pathlib import Path
from datetime import datetime, timezone

def extract_text(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(c.get("text","") for c in content if isinstance(c,dict) and c.get("type")=="text").strip()
    return ""

home = Path.home()
sessions_dir = home / ".claude" / "sessions"
projects_dir = home / ".claude" / "projects"

if not sessions_dir.exists():
    print("[]")
    sys.exit(0)

results = []
for sf in sessions_dir.glob("*.json"):
    try:
        meta = json.loads(sf.read_text())
    except Exception:
        continue

    session_id = meta.get("sessionId", "")
    if not session_id or session_id.startswith("agent-"):
        continue

    pid = meta.get("pid")
    alive = False
    if pid:
        try:
            os.kill(pid, 0)
            alive = True
        except Exception:
            pass

    cwd = meta.get("cwd", "")
    started_at = meta.get("startedAt", 0)

    # Find and parse JSONL
    task_summary = ""
    message_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    cache_creation_tokens = 0
    cache_read_tokens = 0
    last_activity = started_at
    first_msgs = []
    recent_msgs = []
    jsonl_path = None
    if projects_dir.exists():
        for jp in projects_dir.rglob(f"{session_id}.jsonl"):
            jsonl_path = jp
            break

    if jsonl_path:
        try:
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except Exception:
                        continue
                    msg_type = entry.get("type")
                    role = entry.get("message", {}).get("role", msg_type)
                    if role in ("user", "assistant"):
                        message_count += 1
                    if role == "assistant":
                        usage = entry.get("message", {}).get("usage", {})
                        total_input_tokens += usage.get("input_tokens", 0)
                        total_output_tokens += usage.get("output_tokens", 0)
                        cache_creation_tokens += usage.get("cache_creation_input_tokens", 0)
                        cache_read_tokens += usage.get("cache_read_input_tokens", 0)
                    if role in ("user", "assistant"):
                        text = extract_text(entry.get("message", {}).get("content", ""))
                        if text and not text.startswith("<"):
                            me = {"role": role, "text": text[:500]}
                            if len(first_msgs) < 3:
                                first_msgs.append(me)
                            recent_msgs.append(me)
                            if len(recent_msgs) > 6:
                                recent_msgs.pop(0)
                    if not task_summary and msg_type == "user":
                        text = extract_text(entry.get("message", {}).get("content", ""))
                        if text and not text.startswith("<"):
                            task_summary = text[:300]
            mtime_ms = int(jsonl_path.stat().st_mtime * 1000)
            if mtime_ms > last_activity:
                last_activity = mtime_ms
        except Exception:
            pass

    if task_summary and "concise status reporter" in task_summary:
        continue

    excerpt = list(first_msgs)
    first_texts = {m["text"] for m in first_msgs}
    for m in recent_msgs:
        if m["text"] not in first_texts:
            excerpt.append(m)

    started_dt = datetime.fromtimestamp(started_at / 1000, tz=timezone.utc).isoformat() if started_at else ""
    last_dt = ""
    if last_activity and last_activity != started_at:
        last_dt = datetime.fromtimestamp(last_activity / 1000, tz=timezone.utc).isoformat()

    results.append({
        "sessionId": session_id,
        "pid": pid,
        "cwd": cwd,
        "project": os.path.basename(cwd) if cwd else "",
        "startedAt": started_dt,
        "lastActivity": last_dt,
        "kind": meta.get("kind", ""),
        "entrypoint": meta.get("entrypoint", ""),
        "alive": alive,
        "taskSummary": task_summary,
        "messageCount": message_count,
        "tokenUsage": {
            "totalInputTokens": total_input_tokens,
            "totalOutputTokens": total_output_tokens,
            "cacheCreationTokens": cache_creation_tokens,
            "cacheReadTokens": cache_read_tokens,
        },
        "conversationExcerpt": excerpt,
    })

print(json.dumps(results))
'''


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def _collect_all():
    """Collect sessions from all configured sources in parallel."""
    config = load_config()
    all_sessions = []

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {}

        if config.get("include_local", True):
            futures[pool.submit(collect_local)] = "local"

        for srv in config.get("servers", []):
            futures[pool.submit(collect_remote, srv)] = srv.get("label", srv["host"])

        for future in as_completed(futures):
            try:
                result = future.result()
                all_sessions.extend(result)
            except Exception as e:
                label = futures[future]
                all_sessions.append({"server": label, "error": str(e)})

    return all_sessions


def _get_cached():
    """Return cached data or refresh if stale."""
    with _cache["lock"]:
        now = time.time()
        if _cache["data"] is not None and (now - _cache["ts"]) < CACHE_TTL:
            return _cache["data"]

    data = _collect_all()

    with _cache["lock"]:
        _cache["data"] = data
        _cache["ts"] = time.time()

    return data


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    """Force cache invalidation and re-collect."""
    with _cache["lock"]:
        _cache["data"] = None
        _cache["ts"] = 0
    return api_sessions()


# ---------------------------------------------------------------------------
# Server configuration API
# ---------------------------------------------------------------------------

@app.route("/api/config")
def api_config_get():
    """Return current configuration."""
    config = load_config()
    return jsonify(config)


@app.route("/api/config", methods=["PUT"])
def api_config_put():
    """Replace the entire configuration."""
    config = request.get_json()
    if not config:
        return jsonify({"error": "Invalid JSON"}), 400
    save_config(config)
    _invalidate_cache()
    return jsonify(config)


@app.route("/api/servers", methods=["GET"])
def api_servers_list():
    config = load_config()
    servers = config.get("servers", [])
    # Add an index to each server for identification
    for i, s in enumerate(servers):
        s["id"] = i
    return jsonify({
        "servers": servers,
        "include_local": config.get("include_local", True),
    })


@app.route("/api/servers", methods=["POST"])
def api_servers_add():
    """Add a new server."""
    server = request.get_json()
    if not server or not server.get("host"):
        return jsonify({"error": "host is required"}), 400

    entry = {}
    for key in ("host", "port", "user", "key", "label"):
        if server.get(key):
            val = server[key]
            entry[key] = int(val) if key == "port" else val

    config = load_config()
    config.setdefault("servers", []).append(entry)
    save_config(config)
    _invalidate_cache()
    return jsonify({"ok": True, "index": len(config["servers"]) - 1}), 201


@app.route("/api/servers/<int:idx>", methods=["PUT"])
def api_servers_update(idx):
    """Update a server by index."""
    server = request.get_json()
    if not server:
        return jsonify({"error": "Invalid JSON"}), 400

    config = load_config()
    servers = config.get("servers", [])
    if idx < 0 or idx >= len(servers):
        return jsonify({"error": "Server not found"}), 404

    entry = {}
    for key in ("host", "port", "user", "key", "label"):
        if server.get(key):
            val = server[key]
            entry[key] = int(val) if key == "port" else val

    servers[idx] = entry
    config["servers"] = servers
    save_config(config)
    _invalidate_cache()
    return jsonify({"ok": True})


@app.route("/api/servers/<int:idx>", methods=["DELETE"])
def api_servers_delete(idx):
    """Remove a server by index."""
    config = load_config()
    servers = config.get("servers", [])
    if idx < 0 or idx >= len(servers):
        return jsonify({"error": "Server not found"}), 404

    servers.pop(idx)
    config["servers"] = servers
    save_config(config)
    _invalidate_cache()
    return jsonify({"ok": True})


@app.route("/api/config/local", methods=["PUT"])
def api_config_local():
    """Toggle include_local setting."""
    body = request.get_json()
    config = load_config()
    config["include_local"] = bool(body.get("include_local", True))
    save_config(config)
    _invalidate_cache()
    return jsonify({"ok": True, "include_local": config["include_local"]})


@app.route("/api/servers/<int:idx>/test", methods=["POST"])
def api_servers_test(idx):
    """Test SSH connection to a server."""
    config = load_config()
    servers = config.get("servers", [])
    if idx < 0 or idx >= len(servers):
        return jsonify({"error": "Server not found"}), 404

    srv = servers[idx]
    try:
        result = collect_remote(srv)
        has_error = any("error" in r for r in result)
        if has_error:
            return jsonify({"ok": False, "error": result[0].get("error", "Unknown error")})
        return jsonify({"ok": True, "sessions": len(result)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


def _invalidate_cache():
    with _cache["lock"]:
        _cache["data"] = None
        _cache["ts"] = 0


# ---------------------------------------------------------------------------
# AI Summarization via local Claude CLI
# ---------------------------------------------------------------------------

SUMMARIZE_PROMPT = """You are a concise status reporter. Given a conversation excerpt from a Claude Code agent session, produce a brief JSON summary.

The session is working in: {cwd}
The session is currently: {status}

Conversation excerpt:
{conversation}

Respond with ONLY a JSON object (no markdown fencing):
{{
  "task": "<1-sentence: what is the agent working on>",
  "progress": "<1-sentence: current progress/status>",
  "percent": <estimated completion percentage 0-100, or null if unclear>
}}

Use the same language as the conversation (e.g. if the user speaks Chinese, respond in Chinese).
Be specific and concise."""


def _summarize_session(session):
    """Call local claude CLI to summarize a single session."""
    excerpt = session.get("conversationExcerpt", [])
    if not excerpt:
        return None

    conversation_text = "\n".join(
        f"[{m['role']}]: {m['text']}" for m in excerpt
    )

    status = "RUNNING" if session.get("alive") else "COMPLETED"
    prompt = SUMMARIZE_PROMPT.format(
        cwd=session.get("cwd", "unknown"),
        status=status,
        conversation=conversation_text,
    )

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "haiku", prompt],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"[WARN] claude CLI failed for {session.get('sessionId', '?')}: {result.stderr[:200]}")
            return None

        text = result.stdout.strip()
        # Strip markdown fencing if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except subprocess.TimeoutExpired:
        print(f"[WARN] claude CLI timed out for {session.get('sessionId', '?')}")
        return None
    except (json.JSONDecodeError, Exception) as e:
        print(f"[WARN] Summarize failed for {session.get('sessionId', '?')}: {e}")
        return None


_summarize_executor = ThreadPoolExecutor(max_workers=3)
_summarize_in_flight = set()  # session IDs currently being summarized


def _on_summary_done(session_id, message_count, future):
    """Callback when a background summary completes."""
    _summarize_in_flight.discard(session_id)
    try:
        result = future.result()
        if result:
            with _summary_cache["lock"]:
                _summary_cache["data"][session_id] = {
                    "messageCount": message_count,
                    "summary": result,
                }
                _save_summary_cache()
    except Exception as e:
        print(f"[WARN] Summary future failed for {session_id}: {e}")


def _summarize_all(sessions):
    """Apply cached summaries and kick off background summarization for uncached ones.

    Non-blocking: sessions without a cached summary get one on the next refresh.
    """
    for s in sessions:
        if "error" in s:
            continue
        sid = s.get("sessionId", "")
        mc = s.get("messageCount", 0)

        with _summary_cache["lock"]:
            cached = _summary_cache["data"].get(sid)

        if cached and cached.get("messageCount") == mc:
            s["aiSummary"] = cached["summary"]
        else:
            # Keep the old summary visible while regenerating
            if cached:
                s["aiSummary"] = cached["summary"]
            if sid not in _summarize_in_flight and mc > 0:
                _summarize_in_flight.add(sid)
                session_data = {
                    "sessionId": sid,
                    "cwd": s.get("cwd", ""),
                    "alive": s.get("alive", False),
                    "conversationExcerpt": s.get("conversationExcerpt", []),
                }
                fut = _summarize_executor.submit(_summarize_session, session_data)
                fut.add_done_callback(lambda f, sid=sid, mc=mc: _on_summary_done(sid, mc, f))


@app.route("/api/sessions")
def api_sessions():
    # collect_all only returns currently active sessions
    sessions = _get_cached()

    # Run AI summarization
    _summarize_all(sessions)

    # Detect running→completed transitions
    current_running_ids = set()
    running = []
    errors = []

    for s in sessions:
        if "error" in s:
            errors.append(s)
            continue
        sid = s.get("sessionId", "")
        current_running_ids.add(sid)
        # Build a copy without conversationExcerpt for the response
        clean = {k: v for k, v in s.items() if k != "conversationExcerpt"}
        running.append(clean)
        # Track in known_sessions; remove from completed if resumed
        with _known_sessions["lock"]:
            _known_sessions["running"][sid] = clean
            _known_sessions["completed"].pop(sid, None)

    # Sessions that were running before but are no longer → completed
    with _known_sessions["lock"]:
        gone_ids = set(_known_sessions["running"].keys()) - current_running_ids
        for sid in gone_ids:
            s = _known_sessions["running"].pop(sid)
            s["alive"] = False
            _known_sessions["completed"][sid] = s

        completed = list(_known_sessions["completed"].values())

    running.sort(key=lambda x: x.get("startedAt", ""), reverse=True)
    completed.sort(key=lambda x: x.get("lastActivity", "") or x.get("startedAt", ""), reverse=True)

    return jsonify({
        "running": running,
        "completed": completed,
        "errors": errors,
        "collectedAt": datetime.now(timezone.utc).isoformat(),
        "cacheTTL": CACHE_TTL,
    })


@app.route("/api/sessions/<session_id>/dismiss", methods=["POST"])
def api_session_dismiss(session_id):
    """Remove a completed session from the board."""
    with _known_sessions["lock"]:
        if session_id in _known_sessions["completed"]:
            del _known_sessions["completed"][session_id]
            return jsonify({"ok": True})
    return jsonify({"error": "Session not found in completed"}), 404


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Claude Code Agent Kanban")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", "-p", type=int, default=5555, help="Bind port (default: 5555)")
    args = parser.parse_args()

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _load_summary_cache()
    print(f"Claude Code Agent Kanban running at http://localhost:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
