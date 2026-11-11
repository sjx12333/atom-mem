"""Runtime configuration, optional model adapters, and file diagnostics."""

from __future__ import annotations

import json
import logging
import math
import urllib.error
import urllib.request
import subprocess
import time
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from .paths import CONFIG_ROOT, HOME_ROOT, project_relative, resolve_local_path


CONFIG_PATH = CONFIG_ROOT / "graphiti_config.json"
CONFIG_TEMPLATE = CONFIG_ROOT / "graphiti_config.json.example"
_neo4j_process: Optional[subprocess.Popen] = None
_neo4j_start_lock = threading.Lock()
DEFAULT_CONFIG: Dict[str, Any] = {
    "schema_version": 1,
    "database": {
        "neo4j_home": "",
        "neo4j_uri": "",
        "neo4j_user": "",
        "neo4j_password": "",
        "neo4j_database": "neo4j",
        "neo4j_http_port": 7474,
    },
    "embedding": {
        "enabled": True,
        "mode": "cloud",
        "provider": "sentence_transformers",
        "model_path": "data/graphiti/models/embedding",
        "dimension": 1024,
        "device": "auto",
        "model": "text-embedding-v4",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "",
    },
    "rerank": {
        "enabled": True,
        "mode": "cloud",
        "provider": "sentence_transformers",
        "model_path": "data/graphiti/models/rerank",
        "device": "auto",
        "model": "qwen3-rerank",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "",
    },
    "chat": {
        "enabled": True,
        "provider": "openai_compat",
        "base_url": "https://api.deepseek.com",
        "api_key": "",
        "model": "deepseek-v4-flash",
        "temperature": 0.7,
        # Extraction is one JSON answer from a reasoning model: the reasoning text
        # is billed against the same budget, so a tight limit makes the model hit
        # `finish_reason=length` with empty content and the batch publishes
        # nothing. The old 4096 was measurably too small.
        "max_tokens": 8192,
        "timeout": 60,
    },
    "features": {"narrative_orchestration_v2_enabled": False},
    # Extraction cost/shape switches, kept together so an A/B run can flip them
    # without touching code. All are read per extraction, so a restart is enough.
    "extraction_opt": {
        "enabled": True,
        # A second statements call over the same conversation. Its prompt is the
        # first call's prompt plus a suffix, so the platform prefix cache covers
        # almost all of its input; it is the cheapest known way to recover the
        # concrete details (times, places, numbers) a single pass drops.
        "second_pass": True,
        # Skip the per-edge dedupe LLM call when no candidate is lexically close.
        "dedup_gate": True,
        "dedup_gate_min_similarity": 0.62,
        # How many previous episodes are pasted into the extraction prompt. The
        # prompt tells the model to ignore them for extraction, so ten is waste.
        "schema_limit": 3,
    },
    "logging": {"level": "INFO", "retention_days": 30},
    "memory": {
        # SQLite is the online authority; JSON under the authority root is
        # reduced to backup/export/restore.
        "sqlite_path": "data/graphiti/memory.sqlite3",
        "busy_timeout_ms": 10000,
    },
    # The post-conversation memory pass. At session close the AI writes a short
    # first-person diary and the memory nodes are extracted **from that diary**
    # (never from the raw conversation), so what survives is the AI's own view.
    # `key_section` names the chat credential to spend: the pet's live replies use
    # `chat`, so pointing this at `chat_backup` keeps the two from queueing behind
    # each other; it falls back to `chat` when no backup key exists.
    "diary": {
        "enabled": True,
        "memory_extraction": True,
        "key_section": "chat_backup",
    },
    "vector": {
        # Vectors live inside Neo4j next to the objects they describe; there is no
        # separate vector database and no collection path to configure. `dimension`
        # must match the embedding model, and `collection` is only a label used in
        # diagnostics.
        "collection": "graphiti_memory",
        "dimension": 1024,
    },
    # Versioned global retrieval policy (ADR-0034 / ADR-0151). It is read once
    # when a service starts and captured for the session; no role config may
    # override it. The values below are the calibrated defaults: the live
    # reranker separates off-topic passages (<= 0.24) and same-vocabulary
    # passages (<= 0.46) from genuinely relevant ones (>= 0.78), so a zero gate
    # is what let "几乎不相干" content into the prompt.
    "retrieval": {
        "version": "graphiti-recall-v4",
        "semantic_threshold": 0.25,
        "rerank_enabled": True,
        "direct_rerank_threshold": 0.50,
        "indirect_rerank_threshold": 0.55,
        "relative_floor_ratio": 0.40,
        "direct_top_k": 20,
        "token_budget": 8000,
        "semantic_weight": 0.5,
        "rerank_weight": 0.5,
        # 温层召回（三层记忆架构）：近期正文有自己的相似度地板与层内相对地板。
        # 0.15 实测太松（无关句子的余弦普遍 0.2~0.3，整段最近对话都可能被注入），
        # 默认收紧到 0.35 + 0.55×首名；最高分那条永不被相对地板砍掉，所以收紧
        # 不会退化成零召回。召回只读这一层，不再读 ChatLog。
        "warm_similarity_threshold": 0.35,
        "warm_relative_floor_ratio": 0.55,
        "warm_top_k": 8,
        # Chat-log recall (ADR-0163): deprecated since 阶段 6.1. The transcript is
        # still written for the WebUI view and the round audit, but recall reads the
        # Warm Layer instead; these fields are kept so an existing config still
        # parses and are no longer read by `gateway.search()`.
        "chatlog_enabled": False,
        "chatlog_semantic_threshold": 0.15,
        "chatlog_semantic_weight": 0.6,
        "chatlog_recency_weight": 0.4,
        "chatlog_recency_decay": 0.97,
        "chatlog_top_k": 4,
        "chatlog_token_budget": 2000,
        "chatlog_max_entries": 2000,
        # 日历视图（ADR-0178）：时间表达句子的按天侧表。纯规则、不抽取，只作为
        # 召回来源读；命中条目占用 top_k 名额，不额外加塞。
        "calendar_enabled": True,
        "calendar_top_k": 3,
        "calendar_lexical_floor": 0.30,
        # A temp-graph turn hit triggers exactly one batched entity recall
        # (ADR-0164).
        "temp_hit_entity_recall": True,
        "temp_hit_entity_top_k": 5,
        # Graphiti Core 的混合检索默认关闭：它的结果没有标定分数，网关固定全部
        # 丢弃（admitted 0），却要付出 BM25+BFS+community 与逐 passage 云端重排的
        # 代价。留作诊断/实验开关，不参与正式召回。
        "core_search_enabled": False,
        # 重排前先挂少量事实（重排必须看到最终要注入的证据）。
        "preview_items_per_entity": 3,
        # 精准联想提名：hook 的完整短语逐字出现在查询里、且只指向一个母实体时，
        # 允许它把母实体提名为候选；母实体仍要过重排门槛。共享 hook 不提。
        "hook_exact_admission_enabled": True,
        "hook_exact_min_chars": 2,
        "hook_exact_max_candidates": 3,
        # 有界一跳扩展：只用已过门槛的种子，总候选有硬上限。
        "expansion_enabled": True,
        "expansion_max_candidates": 60,
        "expansion_neighbors_per_seed": 5,
        "expansion_hop_decay": 0.70,
        # 只沿首个已过门槛的命中实体走；深度权值默认关闭，实验配置再打开。
        "expansion_seed_count": 1,
        "expansion_depth_bonus": 0.0,
        # ADR-0175：事实是召回单元，实体降级为类别/导航层。生产开启；库默认关闭
        # 以保持既有测试与调用方契约，显式构造 RetrievalPolicy 时可选择新路径。
        "fact_first_enabled": True,
        # Method A: multi-entity query anchoring (entity linking + per-anchor facts).
        "entity_anchoring_enabled": True,
        "anchor_facts_per_entity": 6,
        "anchor_seed_limit": 3,
        "entity_fallback_enabled": True,
        "expansion_strategy": "bfs",
        "expansion_max_depth": 2,
        "expansion_facts_per_entity": 4,
        "fact_near_duplicate_ratio": 0.62,
        "hub_penalty_strength": 0.15,
        # v6：路径证据与文本证据分开。间接候选按 max(重排分, 路径分) 准入，排序分
        # 是 0.65*文本 + 0.35*路径；注入副本走紧凑渲染（事实只留正文）。
        "chain_ranking_weight": 0.35,
        "chain_gate_enabled": True,
        "chain_semantic_floor": 0.18,
        "compact_prompt_content": True,
        "query_routing_enabled": True,
        "rrf_fusion_enabled": True,
        "rrf_k": 60,
        # 激活权重（强化/遗忘）：排序偏好，不改任何准入门槛。
        "activation_enabled": True,
        "activation_rank_bonus": 0.10,
        "activation_edge_weighting": True,
        "activation_weight_floor": 0.25,
        "activation_weight_max": 1.0,
        "activation_initial_weight": 0.5,
        "activation_gain_alpha": 0.40,
        "activation_gain_decay": 0.80,
        "activation_gain_min": 0.04,
        "activation_half_life_hours": 168.0,
        "activation_half_life_growth": 1.2,
        "activation_edge_gain_ratio": 0.5,
    },
    # 热区 v2：近 7 天、累计命中 >=5、至少 2 个批次各自命中率 >30%，只收单条事实。
    "hot_zone": {
        "hit_window_days": 7,
        "min_hit_rounds": 5,
        "min_batch_hit_rate": 0.30,
        "min_hit_batches": 2,
        "batch_rate_metric": "round",
        "include_events": False,
        "max_age_days": 0,
        "max_items_per_anchor": 2,
        "token_ratio": 0.25,
        "settle_quiet_seconds": 3600,
    },
    # 空闲反思：自动生成人设补注（自动通过，用户可手动修改/暂停/回退）。
    "reflection": {
        "enabled": True,
        "interval_seconds": 86400,
        "min_idle_seconds": 3600,
        "backoff_seconds": 3600,
        "max_items": 3,
        "max_overlay_tokens": 400,
        "max_overlay_items": 12,
        "key_section": "chat_backup",
        "auto_activate": True,
    },
}


def _merge(default: Dict[str, Any], value: Any) -> Dict[str, Any]:
    result = dict(default)
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(result.get(key), dict) and isinstance(item, dict):
                result[key] = _merge(result[key], item)
            else:
                result[key] = item
    return result


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists() and CONFIG_TEMPLATE.exists():
        # First run of the standalone package: materialise the documented
        # template so the WebUI has a file to edit.  Credentials are blank in
        # the template and must be filled by the operator.
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(CONFIG_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    if not CONFIG_PATH.exists():
        return validate_config(_merge(DEFAULT_CONFIG, {}))
    try:
        return validate_config(_merge(DEFAULT_CONFIG, json.loads(CONFIG_PATH.read_text(encoding="utf-8"))))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read Graphiti configuration: {exc}") from exc


def load_retrieval_policy(config: Optional[Dict[str, Any]] = None):
    """Build the session's retrieval policy from the versioned global config.

    Imported lazily so this module keeps no import-time dependency on the retrieval
    package. The policy is sanitised by ``RetrievalPolicy.validated`` so a config
    that sets a gate to zero cannot bring back "everything passes".
    """
    from .retrieval import RetrievalPolicy

    source = config if config is not None else load_config()
    section = source.get("retrieval") if isinstance(source, dict) else None
    return RetrievalPolicy.from_value(section if isinstance(section, dict) else {}).validated()



def validate_config(value: Dict[str, Any]) -> Dict[str, Any]:
    config = _merge(DEFAULT_CONFIG, value)
    # One Graphiti extraction chat is the only supported model contract.  Read
    # legacy assignments once for migration, but never retain them in the
    # canonical config written to disk or returned by the API.
    legacy_agents = value.get("agents") if isinstance(value, dict) else None
    legacy_providers = value.get("providers") if isinstance(value, dict) else None
    if isinstance(legacy_agents, dict):
        assignment = legacy_agents.get("main") or {}
        provider = (legacy_providers or {}).get(assignment.get("profile"), {}) if isinstance(legacy_providers, dict) else {}
        if assignment.get("model"):
            config["chat"]["model"] = assignment["model"]
        for key in ("provider", "base_url", "api_key"):
            if provider.get(key):
                config["chat"][key] = provider[key]
    config.pop("agents", None)
    config.pop("providers", None)
    database = config["database"]
    # UI-only projection fields must never become part of the canonical file.
    database.pop("neo4j_password_configured", None)
    # JSON files are the local authority/checkpoint format, not a selectable
    # graph backend. The runtime data root is fixed by the server.
    database.pop("backend", None)
    database.pop("path", None)
    uri = str(database.get("neo4j_uri") or "").strip()
    if uri.lower() == "bolt://localhost":
        database["neo4j_uri"] = "bolt://127.0.0.1:7687"
    elif uri.lower().startswith("bolt://localhost:"):
        database["neo4j_uri"] = "bolt://127.0.0.1:" + uri.split(":", 2)[2]
    if database.get("neo4j_uri") and not str(database.get("neo4j_uri", "")).startswith(("bolt://", "neo4j://", "neo4j+s://")):
        raise ValueError("neo4j URI must start with bolt://, neo4j://, or neo4j+s://")
    for section in ("embedding", "rerank"):
        config[section].pop("api_key_configured", None)
        mode = str(config[section].get("mode", "cpu")).lower()
        if mode not in {"cpu", "cloud"}:
            raise ValueError(f"{section}.mode must be cpu or cloud")
        config[section]["mode"] = mode
        if mode == "cpu":
            model_path = resolve_local_path(config[section].get("model_path", "data/graphiti/models"))
            config[section]["model_path"] = project_relative(model_path)
            if config[section].get("device") not in {"auto", "cpu", "cuda"}:
                raise ValueError(f"{section}.device must be auto, cpu, or cuda")
        else:
            if not str(config[section].get("model", "")).strip():
                raise ValueError(f"{section}.model is required in cloud mode")
            if not str(config[section].get("base_url", "")).strip().startswith(("http://", "https://")):
                raise ValueError(f"{section}.base_url must be an HTTP URL in cloud mode")
        config[section]["dimension"] = int(config[section].get("dimension", 1024)) if section == "embedding" else config[section].get("dimension")
    chat = config["chat"]
    config["chat"].pop("api_key_configured", None)
    if not isinstance(chat.get("enabled"), bool):
        raise ValueError("chat.enabled must be true or false")
    if not str(chat.get("base_url", "")).strip().startswith(("http://", "https://")):
        raise ValueError("chat.base_url must be an HTTP URL")
    if not str(chat.get("model", "")).strip():
        raise ValueError("chat.model is required")
    chat["temperature"] = float(chat.get("temperature", 0.7))
    if not 0 <= chat["temperature"] <= 2:
        raise ValueError("chat.temperature must be between 0 and 2")
    chat["max_tokens"] = int(chat.get("max_tokens", 4096))
    chat["timeout"] = int(chat.get("timeout", 60))
    if chat["max_tokens"] < 1 or chat["timeout"] < 1:
        raise ValueError("chat.max_tokens and chat.timeout must be positive")
    # Optional backup extraction LLM (a different key). Only validated/enforced
    # when it is enabled; an unconfigured disabled block is left as a safe stub.
    backup = config.setdefault("chat_backup", {})
    backup.setdefault("enabled", False)
    backup.setdefault("provider", "openai_compat")
    backup.setdefault("base_url", "")
    backup.setdefault("api_key", "")
    backup.setdefault("model", "")
    backup.setdefault("temperature", 0.7)
    backup.setdefault("max_tokens", 4096)
    backup.setdefault("timeout", 60)
    if backup.get("enabled"):
        if not str(backup.get("model") or "").strip():
            raise ValueError("chat_backup.model is required when chat_backup is enabled")
        base_url = str(backup.get("base_url") or "").strip()
        if base_url and not base_url.startswith(("http://", "https://")):
            raise ValueError("chat_backup.base_url must be an HTTP URL")
        if not str(backup.get("api_key") or "").strip():
            raise ValueError("chat_backup.api_key is required when chat_backup is enabled")
    config["chat_backup"] = backup
    config["database"]["neo4j_http_port"] = int(config["database"].get("neo4j_http_port", 7474))
    if config["logging"].get("level") not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("logging.level is invalid")
    return config


def save_config(value: Dict[str, Any]) -> Dict[str, Any]:
    current = load_config()
    incoming = _merge(current, value)
    # The public projection intentionally omits secrets. Empty values from a
    # reload therefore mean "preserve"; an explicit clear uses
    # ``clear_credentials`` with paths such as ``chat.api_key``.
    clear = set(value.get("clear_credentials", [])) if isinstance(value.get("clear_credentials"), list) else set()
    if isinstance(value.get("database"), dict) and ("database.neo4j_password" not in clear) and ("neo4j_password" not in value["database"] or not value["database"].get("neo4j_password")):
        incoming["database"]["neo4j_password"] = current["database"].get("neo4j_password", "")
    for section in ("embedding", "rerank"):
        if isinstance(value.get(section), dict) and f"{section}.api_key" not in clear and ("api_key" not in value[section] or not value[section].get("api_key")):
            incoming[section]["api_key"] = current[section].get("api_key", "")
    if isinstance(value.get("chat"), dict) and "chat.api_key" not in clear and ("api_key" not in value["chat"] or not value["chat"].get("api_key")):
        incoming["chat"]["api_key"] = current["chat"].get("api_key", "")
    if isinstance(value.get("chat_backup"), dict) and "chat_backup.api_key" not in clear and ("api_key" not in value["chat_backup"] or not value["chat_backup"].get("api_key")):
        incoming["chat_backup"]["api_key"] = current["chat_backup"].get("api_key", "")
    incoming.pop("clear_credentials", None)
    config = validate_config(incoming)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".graphiti_config.", suffix=".tmp", dir=str(CONFIG_PATH.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, CONFIG_PATH)
        try:
            directory_fd = os.open(str(CONFIG_PATH.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return config


def public_config(value: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the complete runtime config for the trusted local WebUI.

    The management route is loopback/session protected. This project uses a
    single-machine WebUI and intentionally follows the reference project's
    plaintext settings editor, so passwords and API keys are not redacted.
    """
    config = validate_config(value or load_config())
    result = json.loads(json.dumps(config))
    for section in ("database", "embedding", "rerank", "chat", "chat_backup"):
        if section in result and isinstance(result[section], dict):
            # Keep secrets out of the WebUI response, but expose an explicit
            # configured flag so password inputs do not misleadingly render
            # as empty/unconfigured after every reload.  The save endpoint
            # treats an empty key as "preserve existing" (see save_config).
            if section != "database":
                secret = result[section].pop("api_key", None)
                result[section]["api_key_configured"] = bool(str(secret or "").strip())
            if section == "database":
                password = result[section].pop("neo4j_password", None)
                result[section]["neo4j_password_configured"] = bool(str(password or "").strip())
    return result


def neo4j_connection_status(value: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Test the configured Bolt connection without bypassing the memory gateway."""
    database = validate_config(value or load_config())["database"]
    uri = str(database.get("neo4j_uri") or "").strip()
    if not uri:
        return {"status": "not_configured", "message": "Neo4j URI 未配置", "uri": ""}
    try:
        from neo4j import GraphDatabase
        # Prefer IPv4 loopback when older configurations still use localhost.
        if uri.lower().startswith("bolt://localhost"):
            uri = uri.replace("bolt://localhost", "bolt://127.0.0.1", 1)
        driver = GraphDatabase.driver(
            uri,
            auth=(str(database.get("neo4j_user") or ""), str(database.get("neo4j_password") or "")),
            connection_timeout=5,
        )
        try:
            driver.verify_connectivity()
            with driver.session(database=str(database.get("neo4j_database") or "neo4j")) as session:
                session.run("RETURN 1 AS ok").single()
        finally:
            driver.close()
        return {"status": "connected", "message": "Neo4j 连接成功", "uri": uri, "database": database.get("neo4j_database", "neo4j")}
    except Exception as exc:
        return {"status": "error", "message": str(exc), "uri": uri, "database": database.get("neo4j_database", "neo4j")}


def neo4j_process_status(value: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the reference-project style local Neo4j lifecycle status."""
    global _neo4j_process
    connection = neo4j_connection_status(value)
    if connection["status"] == "connected":
        return {**connection, "process": "running"}
    if _neo4j_process is not None and _neo4j_process.poll() is None:
        return {**connection, "status": "starting", "process": "starting", "message": "Neo4j 正在启动"}
    return {**connection, "process": "stopped", "status": "error" if connection["status"] == "error" else connection["status"]}


def start_neo4j(value: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Start the configured local Neo4j distribution, then verify Bolt."""
    global _neo4j_process
    current = neo4j_process_status(value)
    if current.get("status") == "connected":
        return current
    database = validate_config(value or load_config())["database"]
    # Neo4j is an installed external runtime (the reference project keeps it
    # at E:\\neo4j); unlike Graphiti data/model assets, its home is allowed
    # outside the checkout.
    home = Path(str(database.get("neo4j_home") or "")).expanduser()
    script = home / "bin" / "neo4j.bat"
    if not script.exists():
        return {"status": "error", "process": "stopped", "message": f"Neo4j 启动脚本不存在: {script}"}
    try:
        # Neo4j's Windows launcher is a PowerShell wrapper.  Calling the batch
        # file through PowerShell (rather than ``Popen(..., shell=True)``)
        # preserves its argument handling and works with Neo4j 5.26.
        ps = os.environ.get("WINDIR", r"C:\Windows") + r"\System32\WindowsPowerShell\v1.0\powershell.exe"
        command = [ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", f"& '{script}' console"]
        _neo4j_process = subprocess.Popen(
            command, cwd=str(script.parent), shell=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            result = neo4j_connection_status(value)
            if result.get("status") == "connected":
                return {**result, "process": "running"}
            if _neo4j_process.poll() is not None:
                break
            time.sleep(1)
        return neo4j_process_status(value)
    except Exception as exc:
        _neo4j_process = None
        return {"status": "error", "process": "stopped", "message": str(exc)}


def stop_neo4j(value: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Stop only the Neo4j process started by this Graphiti gateway."""
    global _neo4j_process
    if _neo4j_process is None:
        return {"status": "stopped", "process": "stopped", "message": "没有由当前网关启动的 Neo4j 进程"}
    try:
        if os.name == "nt" and _neo4j_process.poll() is None:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(_neo4j_process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        else:
            _neo4j_process.terminate()
        _neo4j_process.wait(timeout=15)
    except Exception:
        try:
            _neo4j_process.kill()
        except Exception:
            pass
    finally:
        _neo4j_process = None
    return {"status": "stopped", "process": "stopped", "message": "Neo4j 已停止"}


def configure_logging(root: Path) -> Dict[str, str]:
    """Create a per-process text log and structured event log below Graphiti."""
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    text_path = log_dir / f"graphiti_{run_id}.log"
    event_path = log_dir / f"events_{run_id}.jsonl"
    logger = logging.getLogger("graphiti")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(getattr(handler, "graphiti_run", False) for handler in logger.handlers):
        handler = logging.FileHandler(text_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        handler.graphiti_run = True
        logger.addHandler(handler)
    logger.info("Graphiti run started run_id=%s data_root=%s", run_id, project_relative(root))
    return {"run_id": run_id, "log_file": project_relative(text_path), "event_log": project_relative(event_path)}


def event(
    log_info: Dict[str, str],
    action: str,
    *,
    status: str = "info",
    message: str = "",
    details: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[float] = None,
) -> None:
    """Append one structured event.

    ``duration_ms`` (optional) is the server-side cost of the stage, so the text
    log line answers "这一步花了多久" without opening the JSONL. It is kept as a
    separate top-level key so existing consumers of ``details`` are unaffected.
    """
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": log_info["run_id"],
        "action": action,
        "status": status,
        "message": message,
        "details": details or {},
    }
    suffix = ""
    if duration_ms is not None:
        try:
            payload["duration_ms"] = round(float(duration_ms), 1)
            suffix = f" duration_ms={payload['duration_ms']}"
        except (TypeError, ValueError):
            pass
    with (HOME_ROOT / log_info["event_log"]).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    getattr(logging.getLogger("graphiti"), "error" if status == "failed" else "info")("%s: %s%s", action, message, suffix)


def _device(config: Dict[str, Any]) -> str:
    requested = config.get("device", "auto")
    if requested != "auto":
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _post_json(url: str, payload: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _embedding_url(base_url: str) -> str:
    return base_url.rstrip("/") if base_url.rstrip("/").endswith("/embeddings") else f"{base_url.rstrip('/')}/embeddings"


def _rerank_url(base_url: str) -> str:
    """Resolve the rerank endpoint for a configured base URL.

    Rerank only exists on the DashScope native service path, and this adapter
    speaks the native payload/response protocol, so an OpenAI-compatible base
    (the value the runtime WebUI pre-fills, ``…/compatible-mode/v1``) must be
    reduced to its host before the service path is appended.
    """
    # ---- 旧实现（已停用，原样保留以便回退） ------------------------------
    # 直接把配置里的 base_url 当作服务根，只对 "/services/rerank/" 做特判。
    # 当 base_url 是 DashScope 的 OpenAI 兼容基址（WebUI 默认值）时会拼出
    #   …/compatible-mode/v1/api/v1/services/rerank/text-rerank/text-rerank
    # → 每次调用都 404，项目召回整层降级为 0 条图记忆（详见 ADR-0141 之后的排查记录）。
    # def _rerank_url(base_url: str) -> str:
    #     if "/services/rerank/" in base_url:
    #         return base_url.rstrip("/")
    #     return f"{base_url.rstrip('/')}/api/v1/services/rerank/text-rerank/text-rerank"

    # ---- 兼容实现（当前生效） --------------------------------------------
    # 兼容基址 …/compatible-mode/v1 对 /embeddings 有效，但 DashScope 兼容面没有
    # 任何 rerank 路由（实测 /reranks 与原生路径都 404），因此先归一化到主机+原生路径。
    base = str(base_url or "").strip().rstrip("/")
    if "/services/rerank/" in base:
        return base
    if "/compatible-mode/" in base:
        base = base.split("/compatible-mode/", 1)[0]
    return f"{base}/api/v1/services/rerank/text-rerank/text-rerank"


class _CloudReranker:
    """DashScope rerank adapter that sends **one request per batch**.

    The previous adapter built a request with ``documents: [content]`` and the recall
    layer called it once per candidate, so recall cost ``162 ms × candidates``
    (measured; the candidate cap is 80, i.e. up to ~13 s). Measured on qwen3-rerank,
    the request overhead dominates: 1 document 162 ms, 5 documents 174 ms,
    100 documents 250 ms — and a document's score is independent of its position in
    the batch (a 50-document batch reproduced the single-call score to 16 digits).

    The response sorts results by score and carries each one's ``index``, so scores
    are mapped back to input positions rather than zipped in order.
    """

    def __init__(self, base_url: str, model: str, api_key: str):
        self.endpoint = _rerank_url(base_url)
        self.model = model
        self.api_key = api_key

    def rerank_many(self, query: str, contents) -> List[float]:
        documents = [str(item) for item in contents]
        if not documents:
            return []
        data = _post_json(
            self.endpoint,
            {"model": self.model, "input": {"query": str(query), "documents": documents}},
            self.api_key,
        )
        results = (data.get("output") or {}).get("results") or []
        scores = [0.0] * len(documents)
        for item in results:
            try:
                index = int(item.get("index", 0))
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(scores):
                scores[index] = max(0.0, float(item.get("relevance_score", 0.0) or 0.0))
        return scores

    def __call__(self, query: str, content: str) -> float:
        return self.rerank_many(query, [content])[0]


class _LocalReranker:
    """CrossEncoder adapter with the same batch interface as the cloud one."""

    def __init__(self, model):
        self.model = model

    def rerank_many(self, query: str, contents) -> List[float]:
        documents = [str(item) for item in contents]
        if not documents:
            return []
        pairs = [(str(query), item) for item in documents]
        return [max(0.0, float(value)) for value in self.model.predict(pairs)]

    def __call__(self, query: str, content: str) -> float:
        return self.rerank_many(query, [content])[0]


def load_model_adapters(config: Dict[str, Any]) -> tuple[Optional[Callable[[str], Sequence[float]]], Optional[Callable[..., float]], Dict[str, Any]]:
    """Load only Graphiti-configured local models; never use legacy config."""
    status: Dict[str, Any] = {"embedding": {"status": "disabled"}, "rerank": {"status": "disabled"}}
    embedder = None
    reranker = None
    embedding = config["embedding"]
    if embedding.get("enabled"):
        if embedding.get("mode") == "cloud":
            if not embedding.get("api_key"):
                status["embedding"] = {"status": "missing_config", "mode": "cloud", "model": embedding.get("model")}
            else:
                def embed_cloud(text: str):
                    data = _post_json(_embedding_url(embedding["base_url"]), {"model": embedding["model"], "input": [str(text)]}, embedding["api_key"])
                    return ((data.get("data") or [{}])[0]).get("embedding") or []
                embedder = embed_cloud
                # endpoint 是新增诊断键（旧实现只有 status/mode/model），用于暴露
                # 解析后的真实端点，避免 "configured" 掩盖错误的 base_url。
                status["embedding"] = {"status": "configured", "mode": "cloud", "model": embedding.get("model"), "endpoint": _embedding_url(embedding["base_url"])}
        else:
            path = resolve_local_path(embedding["model_path"])
            if not path.exists():
                status["embedding"] = {"status": "missing", "mode": "cpu", "path": project_relative(path)}
            else:
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(str(path), device=_device(embedding))
                    embedder = lambda text: model.encode(str(text), normalize_embeddings=True).tolist()
                    status["embedding"] = {"status": "loaded", "mode": "cpu", "path": project_relative(path), "dimension": embedding.get("dimension")}
                except Exception as exc:  # optional dependency/model errors must not break lexical recall
                    status["embedding"] = {"status": "error", "mode": "cpu", "path": project_relative(path), "error": str(exc)}
    rerank = config["rerank"]
    if rerank.get("enabled"):
        if rerank.get("mode") == "cloud":
            if not rerank.get("api_key"):
                status["rerank"] = {"status": "missing_config", "mode": "cloud", "model": rerank.get("model")}
            else:
                reranker = _CloudReranker(rerank["base_url"], rerank["model"], rerank["api_key"])
                # endpoint 同上：新增诊断键（旧实现只有 status/mode/model）。
                status["rerank"] = {"status": "configured", "mode": "cloud", "model": rerank.get("model"), "endpoint": _rerank_url(rerank["base_url"])}
        else:
            path = resolve_local_path(rerank["model_path"])
            if not path.exists():
                status["rerank"] = {"status": "missing", "mode": "cpu", "path": project_relative(path)}
            else:
                try:
                    from sentence_transformers import CrossEncoder
                    model = CrossEncoder(str(path), device=_device(rerank))
                    reranker = _LocalReranker(model)
                    status["rerank"] = {"status": "loaded", "mode": "cpu", "path": project_relative(path)}
                except Exception as exc:
                    status["rerank"] = {"status": "error", "mode": "cpu", "path": project_relative(path), "error": str(exc)}
    return embedder, reranker, status
