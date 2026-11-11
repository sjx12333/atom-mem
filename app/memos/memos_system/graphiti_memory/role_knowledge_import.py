"""给**已存在**的角色补充世界知识（多份 txt，各自独立、可单独撤销）。

这与 ``role_import``（文档创建角色）是同构的抽取通路，但边界完全不同：

| | ``role_import``（文档创建角色） | 本模块（补充世界知识） |
| --- | --- | --- |
| 目标角色 | **新建**一个 role_uuid，抽取期占位名，确认后才赋名激活 | **已有的**活动角色，身份节点早已有名 |
| 输入 | 一份原文 | 随便几份 txt，**每份一个独立子任务** |
| 提示词 | 模型生成草稿，用户确认 | 不动角色的提示词 |
| 失败影响 | 整个创建失败 | 只影响这一份文件 |
| 撤销粒度 | 整个未确认分区 | **按这一份文件的账本**撤销 |

复用而不重写的部分：``split_document`` 分块与缩块重试、``run_graphiti_extraction``
低层抽取、``ExtractionPipeline.publish`` → ``PublicationSaga`` 的知识批发布、
``publish_import_memories`` 的记忆批发布、``build_source_memory_prompt`` 的记忆提示词。

**知识去向**：抽出的实体与陈述作为知识节点/事实节点进入**该角色自己的白名单**，
靠语义召回在相关时才注入提示词，不常驻占用每轮 token。

**多次累加**：同一个角色可以反复导入。每份文件按 ``sha256`` 记住，同一份原文重复
导入会被识别出来（``duplicate_of``），不会被静默抽两遍。

**撤销边界**：一次导入把对象加进角色白名单，所以撤销只做"这个角色看不见这批对象"
（``forget_role_node`` / ``forget_role_relation``），**不删除共享图谱对象** ——
别的角色可能也在用同一个实体。这与"删除角色"的 ``purge_role`` 是不同的粒度。
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import uuid
from typing import Any, Sequence

from .coordinator import EXTRACTION_STAGES
from .document_chunking import bisect_document_chunk, split_document
from .domain import typed_id, utc_timestamp
from .identity import normalize_source_identity
from .memory_extraction import (
    MAX_MEMORIES_PER_SOURCE_BLOCK,
    build_source_memory_prompt,
    normalize_memories,
)
from .role_import import MEMORY_MIN_TOKENS, SourceMemoryBatch, _extract_chunk

#: 进度阶段。与 ``role_import.IMPORT_STAGES`` 同源，只是没有"生成人设提示词"
#: ——已有角色的提示词不因为这个功能改变。
KNOWLEDGE_IMPORT_STAGES = [
    *EXTRACTION_STAGES[:4],
    ("extract_statements", "抽取实体陈述"),
    *EXTRACTION_STAGES[4:],
    ("extract_memory", "抽取记忆"),
    ("publish_memory", "写入记忆节点"),
]

#: 一份文件最多递归拆几次（与文档创建角色一致：截断就从中部句界对半拆）。
MAX_SPLIT_DEPTH = 6

#: 小于这个字数不再拆，直接作为失败上报。
MIN_SPLITTABLE_CHARS = 128

#: 记忆批默认开启（用户选定），但可以逐次关掉。
DEFAULT_EXTRACT_MEMORIES = True


def _logical_key(group_id: str, file_index: int, kind: str) -> str:
    """一次导入里"一份文件的一批"的逻辑标识，也是这一批的 ``conversation_id``。

    它会被拼进角色目录下的文件名（批量/归档记账都以它命名），所以**不能含冒号**
    ——Windows 上 ``:`` 是非法路径字符，``.json`` 落盘会直接 ``OSError``。
    用 ``group_id`` 而不是角色 UUID：同一次导入里的文件按序号区分，不同导入之间
    由 group_id 区分，这样两份文件内容相似也不会撞在同一批上。
    """
    return f"knowledge-{group_id}-{file_index}-{kind}"


def _episode_uuid(group_id: str, file_index: int, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"knowledge-import:{group_id}:{file_index}:{chunk_index}"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RoleKnowledgeImportCoordinator:
    """多文件世界知识导入：按文件独立分块、发布、记账与撤销。"""

    def __init__(self, service):
        self.service = service
        self.root = service.root / "role_knowledge_imports"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._load_jobs()

    # ------------------------------------------------------------------ 持久化

    def _load_jobs(self) -> None:
        for path in self.root.glob("*.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(job, dict) or not job.get("job_id"):
                continue
            if job.get("status") in {"queued", "extracting", "publishing"}:
                # 服务重启时线程没了。留着"抽取中"会比报错更糟：界面会永远等下去。
                job.update(
                    status="failed", error_code="IMPORT_INTERRUPTED",
                    error="服务重启，这次补充知识被中断；可以重新导入同一份文件。",
                    message="已中断", updated_at=utc_timestamp(),
                )
                self._save(job)
            self._jobs[job["job_id"]] = job

    def _save(self, job: dict[str, Any]) -> None:
        self.service.authority.save_root_file(f"role_knowledge_imports/{job['job_id']}.json", job)

    def _update(self, job: dict[str, Any], *, log: bool = False, **changes: Any) -> None:
        with self._lock:
            job.update(changes, updated_at=utc_timestamp())
            if log and changes.get("message"):
                job.setdefault("progress_log", []).append(
                    {"timestamp": job["updated_at"], "message": job["message"]}
                )
                job["progress_log"] = job["progress_log"][-100:]
            self._save(job)

    def _warn(self, job: dict[str, Any], message: str, **extra: Any) -> None:
        """记一条非致命问题；``warning`` 只保留最后一条，``warnings`` 累积。"""
        self._update(
            job, log=True, warning=message,
            warnings=[*(job.get("warnings") or []), {"message": message, **extra}][-50:],
        )

    def _memory_budget(self) -> int:
        chat = (getattr(self.service, "runtime_config", {}) or {}).get("chat") or {}
        try:
            configured = int(chat.get("max_tokens") or 4096)
        except (TypeError, ValueError):
            configured = 4096
        return max(MEMORY_MIN_TOKENS, min(8192, configured))

    @staticmethod
    def _stage_index(stage: str) -> int:
        return next((i for i, item in enumerate(KNOWLEDGE_IMPORT_STAGES) if item[0] == stage), 0)

    def _progress(self, job: dict[str, Any], stage: str, message: str, completed: int) -> None:
        self._update(job, log=True, message=message, progress={
            "stages": [{"key": key, "label": label} for key, label in KNOWLEDGE_IMPORT_STAGES],
            "current_stage": stage, "stage_index": self._stage_index(stage),
            "completed": completed, "total": len(KNOWLEDGE_IMPORT_STAGES), "done": False,
        })

    # ------------------------------------------------------------------ 取角色配置

    def _role_context(self, role_uuid: str, *, user_appellation: str = "") -> dict[str, Any]:
        """角色自己的模型配置与身份，用于抽取。

        抽取模型是**角色自己的** provider/model/api_key（``config.json``），不是在
        这里重新问用户一遍：同一个角色补知识，用哪个模型是它已经定好的运行配置。

        ``identity.json`` 只有**文档创建角色**那条通路会写。直接在向导里手写提示词
        建出来的角色没有它，但本功能必须对任何已有角色都能用（补充知识可以反复进行），
        所以缺映射时退回"角色显示名 = 原文里的角色、称呼 = 原文里的用户"这个默认：
        那正是手写提示词的角色在设定里通常用的两个名字，用户改起来也就是改这两个。
        """
        config = self.service.authority.load_role_file(role_uuid, "config.json", {})
        if not isinstance(config, dict) or not config:
            raise ValueError("角色不存在或配置缺失，无法补充知识")
        identity = self.service.authority.load_role_file(role_uuid, "identity.json", {})
        identity = identity if isinstance(identity, dict) else {}
        display_name = str(config.get("display_name") or role_uuid)
        model = str(config.get("model") or "").strip()
        if not model:
            raise ValueError("这个角色没有配置模型，无法抽取知识；请先在角色编辑里补全模型配置")
        appellation = str(
            user_appellation or identity.get("user_appellation")
            or getattr(self.service.roles, "USER_NODE_NAME", "主人")
        )
        return {
            "role_uuid": role_uuid,
            "display_name": display_name,
            "provider": str(config.get("provider") or "openai_compat"),
            "base_url": config.get("base_url"),
            "model": model,
            "api_key": str(config.get("api_key") or ""),
            # 原文里的角色名/用户名沿用角色自己的身份映射（ADR-0167）；缺映射时退回
            # 默认名字，而不是拒绝整个导入。
            "source_role_name": str(identity.get("source_role_name") or display_name),
            "source_user_name": str(identity.get("source_user_name") or "").strip() or appellation,
            "user_appellation": appellation,
        }

    # ------------------------------------------------------------------ 启动

    def start(
        self,
        *,
        role_uuid: str,
        files: Sequence[dict[str, Any]],
        user_appellation: str = "",
        extract_memories: bool = DEFAULT_EXTRACT_MEMORIES,
    ) -> dict[str, Any]:
        """为已有角色补充世界知识。``files`` 是 ``[{name, content}, ...]``。

        每份文件是一个独立子任务：独立分块、独立记账、独立撤销，也是并行推进的单位。
        """
        role = str(role_uuid or "").strip()
        if not role:
            raise ValueError("role_uuid is required")
        self.assert_role_importable(role)
        documents = []
        for item in files or []:
            name = str((item or {}).get("name") or "").strip() or "未命名.txt"
            content = str((item or {}).get("content") or "")
            if not content.strip():
                raise ValueError(f"文件「{name}」是空的，没有可抽取的内容")
            documents.append({"name": name, "content": content})
        if not documents:
            raise ValueError("至少要选择一份 txt 才能补充知识")

        context = self._role_context(role, user_appellation=user_appellation)
        try:
            spec = normalize_source_identity(context)
        except ValueError as exc:
            raise ValueError(
                f"{exc}；这个角色没有可用的原文身份映射，请先在角色编辑里填写原文角色名/用户名"
            ) from exc

        group_id = typed_id("operation")
        with self._lock:
            known = self._fingerprints_for(role)
        for file_index, document in enumerate(documents):
            digest = _sha256(document["content"])
            previous = known.get(digest)
            job_id = typed_id("operation")
            job = {
                "job_id": job_id, "group_id": group_id, "role_uuid": role,
                "file_index": file_index,
                "status": "queued", "message": "等待抽取",
                "file": {
                    "name": document["name"], "sha256": digest,
                    "chars": len(document["content"]),
                },
                # 同一份原文在这个角色上已经导过：照旧执行（用户可能就是想再来一次），
                # 但把上一次的任务 id 记下来，界面能提示"重复"。
                "duplicate_of": previous,
                "options": {"extract_memories": bool(extract_memories)},
                "completed_chunks": 0, "completed_chars": 0, "current_chunk": 0,
                "total_chunks": 0, "retry_count": 0, "memory_candidates": 0,
                "memory_published": 0, "memory_batches": [],
                "published_objects": [], "knowledge_operation_ids": [],
                "progress": {}, "progress_log": [], "created_at": utc_timestamp(),
                "updated_at": utc_timestamp(), "error": None, "error_code": None,
                "warnings": [], "rolled_back": False,
                "spec": {k: v for k, v in spec.items() if k != "api_key"},
            }
            with self._lock:
                self._jobs[job_id] = job
                self._save(job)
            threading.Thread(
                target=self._run,
                args=(job_id, document["content"], spec),
                daemon=True,
            ).start()
        return {
            "group_id": group_id,
            "role_uuid": role,
            "jobs": self.list_jobs(role_uuid=role, group_id=group_id),
        }

    def assert_role_importable(self, role_uuid: str) -> dict[str, Any]:
        """只允许给"能启动的角色"补知识。

        ``initializing``（还没确认的文档创建分区）与 ``deleting`` 都不该被写入；
        被 ``blocked`` 的角色例外——补知识正是修复它的一种手段。
        """
        role = self.service.roles.safe_config(role_uuid)
        lifecycle = str(role.get("lifecycle") or "")
        if lifecycle in {"initializing", "deleting"}:
            raise ValueError(f"角色当前状态是 {lifecycle}，不能补充知识")
        return role

    def _fingerprints_for(self, role_uuid: str) -> dict[str, str]:
        """``{sha256: job_id}``：这个角色已经导过的原文。"""
        known: dict[str, str] = {}
        for job in self._jobs.values():
            if job.get("role_uuid") != role_uuid or job.get("rolled_back"):
                continue
            if job.get("status") == "cancelled":
                continue
            digest = str((job.get("file") or {}).get("sha256") or "")
            if digest:
                known.setdefault(digest, str(job.get("job_id")))
        return known

    # ------------------------------------------------------------------ 执行

    def _run(self, job_id: str, content: str, spec: dict[str, Any]) -> None:
        job = self._jobs[job_id]
        client = None
        try:
            runtime = dict(self.service.runtime_config)
            runtime["chat"] = {
                **runtime.get("chat", {}),
                **{k: spec.get(k) for k in ("provider", "base_url", "model", "api_key")},
            }
            # 与文档创建角色同一套分块预算：输入 token 上限跟着配置的输出预算走。
            budget = max(128, min(2000, int(runtime["chat"].get("max_tokens") or 4096) // 2))
            chunks = [{**chunk, "depth": 0} for chunk in split_document(content, max_tokens=budget)]
            self._update(
                job, log=True, status="extracting", total_chunks=len(chunks),
                message=f"正在抽取「{job['file']['name']}」",
            )
            from graphiti_factory import create_graphiti_client

            client = create_graphiti_client(runtime)
            index = 0
            while index < len(chunks):
                chunk = chunks[index]
                self._update(job, current_chunk=index + 1, status="extracting")
                knowledge_key = _logical_key(job["group_id"], index, "k")
                snapshot = {
                    "role_uuid": job["role_uuid"],
                    "conversation_id": knowledge_key,
                    "user_appellation": spec["user_appellation"],
                    "source_role_name": spec["source_role_name"],
                    "source_user_name": spec["source_user_name"],
                    # 身份节点已经是正式名字（角色早就存在），不需要延后赋名。
                    "identity_name_pending": False,
                    "events": [{
                        "event_id": f"{job_id}-{index}", "role": "user",
                        "content": chunk["target_text"], "complete": True, "observable": True,
                    }],
                }
                progress = lambda stage, message, completed: self._progress(job, stage, message, completed)
                try:
                    pipeline_result = self.service._run_graphiti_async(_extract_chunk(
                        client, _episode_uuid(job["group_id"], int(job["file_index"]), index),
                        chunk["target_text"], job["role_uuid"], knowledge_key,
                        progress=progress,
                        custom_extraction_instructions=self.service._extraction_instructions(
                            snapshot, str(spec["user_appellation"]), str(spec["display_name"])
                        ),
                    ))
                except Exception as exc:  # noqa: BLE001 - 分类在下面
                    from graphiti_core.llm_client.errors import OutputTruncatedError

                    if not isinstance(exc, OutputTruncatedError):
                        raise
                    text = chunk["target_text"]
                    if chunk["depth"] >= MAX_SPLIT_DEPTH or len(text) < MIN_SPLITTABLE_CHARS:
                        raise
                    chunks[index:index + 1] = [
                        {"target_text": part, "depth": chunk["depth"] + 1}
                        for part in bisect_document_chunk(text)
                    ]
                    self._update(
                        job, log=True, retry_count=job["retry_count"] + 1,
                        total_chunks=len(chunks),
                        message=f"第 {index + 1} 块输出达到上限，已拆成两块重试",
                    )
                    continue

                from .extraction_engine import build_extracted_payload

                # ``raw_text`` 必须带上，否则知识层的比喻/喻体门槛会失效。
                extracted = build_extracted_payload({**pipeline_result, "raw_text": chunk["target_text"]})
                self._update(job, status="publishing")
                published = self.service.pipeline.publish(
                    snapshot, lambda _text, value=extracted: value, progress=progress
                )
                self.service.pipeline.cleanup(snapshot)
                self._record_publication(job, published, kind="knowledge", chunk_index=index)

                if job["options"].get("extract_memories"):
                    self._progress(job, "extract_memory", "抽取这个角色自己的经历", self._stage_index("extract_memory"))
                    memory_extract = self._extract_memories(job, client, chunk["target_text"], spec)
                    memories = memory_extract["memories"]
                    if memory_extract.get("error"):
                        self._warn(
                            job, f"第 {index + 1} 块的记忆未抽出：{memory_extract['error']}",
                            chunk_index=index, error=memory_extract["error"],
                            truncated=bool(memory_extract.get("truncated")),
                        )
                    memory_key = _logical_key(job["group_id"], index, "m")
                    memory_result: dict[str, Any] = {"status": "empty", "published_ids": [], "candidates": 0}
                    if memories:
                        self._progress(job, "publish_memory", "写入记忆节点", self._stage_index("publish_memory"))
                        try:
                            memory_result = self.service.publish_import_memories(
                                role_uuid=job["role_uuid"], conversation_id=memory_key,
                                memories=memories, appellation=str(spec["user_appellation"]),
                                display_name=str(spec["display_name"]),
                                source_role_name=str(spec["source_role_name"]),
                                source_user_name=str(spec["source_user_name"]),
                            )
                        except Exception as exc:  # noqa: BLE001 - 记忆是尽力而为，知识不受影响
                            memory_result = {
                                "status": "failed", "published_ids": [],
                                "candidates": len(memories), "error": f"{type(exc).__name__}: {exc}",
                            }
                    self._update(
                        job,
                        memory_candidates=job["memory_candidates"] + int(memory_result.get("candidates") or 0),
                        memory_published=job["memory_published"] + len(memory_result.get("published_ids") or []),
                        published_objects=[
                            *job["published_objects"],
                            *[str(item) for item in (memory_result.get("published_ids") or [])],
                        ][-5000:],
                        memory_batches=[*job["memory_batches"], {
                            "chunk_index": index, "conversation_id": memory_key,
                            "status": "failed" if memory_extract.get("error") else memory_result.get("status"),
                            "candidates": memory_result.get("candidates", 0),
                            "published": len(memory_result.get("published_ids") or []),
                            "published_ids": [str(item) for item in (memory_result.get("published_ids") or [])],
                            "operation_id": memory_result.get("operation_id"),
                            "error": memory_extract.get("error") or memory_result.get("error"),
                            "truncated": bool(memory_extract.get("truncated")),
                        }][-200:],
                    )

                self._update(
                    job, log=True, completed_chunks=index + 1,
                    completed_chars=job["completed_chars"] + len(chunk["target_text"]),
                    message=f"已完成 {index + 1}/{len(chunks)} 块",
                )
                index += 1

            self._update(
                job, log=True, status="completed",
                progress={**job["progress"], "done": True, "completed": len(KNOWLEDGE_IMPORT_STAGES)},
                message=f"「{job['file']['name']}」的知识已写入，下一次会话起可被召回",
                finished_at=utc_timestamp(),
            )
        except Exception as exc:  # noqa: BLE001 - 全部转成任务状态
            from graphiti_core.llm_client.errors import OutputTruncatedError

            if isinstance(exc, OutputTruncatedError):
                error = (
                    "模型输出达到上限，缩小文本块后仍未完成。请提高这个角色的模型最大输出 Token，"
                    "或改用输出预算更充足的模型。"
                )
                code = "LLM_OUTPUT_TRUNCATED"
            else:
                error = str(exc)
                for secret in (spec.get("api_key"), spec.get("base_url")):
                    if secret:
                        error = error.replace(str(secret), "[已隐藏]")
                code = type(exc).__name__
            self._update(job, log=True, status="failed", message="补充知识失败",
                         error=error, error_code=code, finished_at=utc_timestamp())
        finally:
            if client is not None:
                try:
                    self.service._run_graphiti_async(client.close())
                except Exception:  # noqa: BLE001 - 关连接失败不能推翻已发布的知识
                    self._warn(job, "抽取连接释放失败；已保存的进度和结果不受影响")

    def _record_publication(
        self, job: dict[str, Any], published: dict[str, Any], *, kind: str, chunk_index: int
    ) -> None:
        """把这一批真正写出的对象 id 记进任务，作为撤销的账本。"""
        manifest = ((published or {}).get("manifest") or {}) if isinstance(published, dict) else {}
        ids = [str(item) for item in (manifest.get("published_ids") or []) if item]
        self._update(
            job,
            knowledge_operation_ids=[
                *job["knowledge_operation_ids"],
                {"chunk_index": chunk_index, "kind": kind, "object_ids": ids},
            ][-500:],
            published_objects=[*job["published_objects"], *ids][-5000:],
        )

    def _extract_memories(
        self, job: dict[str, Any], client: Any, text: str, spec: dict[str, Any]
    ) -> dict[str, Any]:
        """这一块的记忆批：这个角色自己经历过的事。失败只记警告，不回滚知识。"""
        report: dict[str, Any] = {"memories": [], "attempts": 0, "error": None, "truncated": False}
        if client is None or not str(text or "").strip():
            return report
        report["memories"] = self._memory_batch(client, str(text), spec, depth=0, report=report)
        return report

    def _memory_batch(
        self, client: Any, text: str, spec: dict[str, Any], *, depth: int, report: dict[str, Any]
    ) -> list[dict[str, Any]]:
        from graphiti_core.llm_client.errors import OutputTruncatedError
        from graphiti_core.prompts.models import Message

        report["attempts"] = int(report.get("attempts") or 0) + 1
        try:
            result = self.service._run_graphiti_async(client.llm_client.generate_response([
                Message(role="system", content="你只输出 JSON。"),
                Message(role="user", content=build_source_memory_prompt(
                    source=text,
                    appellation=str(spec.get("user_appellation") or ""),
                    display_name=str(spec.get("display_name") or ""),
                    source_role_name=str(spec.get("source_role_name") or ""),
                )),
            ], response_model=SourceMemoryBatch, max_tokens=self._memory_budget()))
            payload = result.model_dump() if hasattr(result, "model_dump") else dict(result or {})
            return normalize_memories(payload.get("memories") or [], limit=MAX_MEMORIES_PER_SOURCE_BLOCK)
        except OutputTruncatedError as exc:
            report["truncated"] = True
            if depth < 1 and len(text) >= 400:
                memories: list[dict[str, Any]] = []
                for part in bisect_document_chunk(text):
                    memories.extend(self._memory_batch(client, part, spec, depth=depth + 1, report=report))
                return memories
            report["error"] = (
                f"记忆抽取输出达到上限（{exc}）：这一段的记忆没有写入。"
                "可在设置里提高抽取模型的最大输出 Token。"
            )
            return []
        except Exception as exc:  # noqa: BLE001 - 记忆是尽力而为
            report["error"] = f"{type(exc).__name__}: {exc}"
            return []

    # ------------------------------------------------------------------ 查询

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                raise KeyError(job_id)
            return self.public(job)

    def list_jobs(
        self, role_uuid: str | None = None, group_id: str | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [
                self.public(job) for job in self._jobs.values()
                if (not role_uuid or job.get("role_uuid") == role_uuid)
                and (not group_id or job.get("group_id") == group_id)
            ]
        return sorted(jobs, key=lambda job: job.get("created_at", ""), reverse=True)

    # ------------------------------------------------------------------ 撤销

    def rollback(self, job_id: str) -> dict[str, Any]:
        """撤销**这一次导入**：让这个角色看不见这批对象。

        撤销的粒度是"这个角色 + 这批对象"，不是"删掉这些图谱对象"：知识节点是共享的，
        另一个角色可能已经通过它自己的导入认识同一个实体，删掉就是替别人做决定。
        所以这里只撤销白名单并给这个角色打遗忘墓碑（``forget_role_node`` /
        ``forget_role_relation``，即角色遗忘的既有语义）。

        已经和别的导入共用的对象会被一并遗忘 —— 这是显式接受的取舍：撤销是"我不再
        记得这批知识"，而图谱不记录"这条描述是哪一次导入加进去的"。报告里会列出
        被遗忘的对象，供用户判断是否要重新导入其它文件。
        """
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        if job.get("status") in {"queued", "extracting", "publishing"}:
            raise ValueError("这次导入还在进行中，请等它结束再撤销")
        if job.get("rolled_back"):
            raise ValueError("这次导入已经撤销过了")

        role_uuid = job["role_uuid"]
        ledger = [str(item) for item in (job.get("published_objects") or []) if item]
        memory_ids = [
            str(item)
            for entry in job.get("memory_batches") or []
            if entry.get("status") == "published"
            for item in (entry.get("published_ids") or [])
        ]
        forgotten: list[str] = []
        skipped: list[str] = []
        errors: list[dict[str, Any]] = []
        for target in dict.fromkeys([*ledger, *memory_ids]):
            # 事实节点没有自己的白名单条目：它挂在端点实体上，随实体一起不可见。
            # 单独去 revoke 它只会产生一条噪音墓碑，所以只报"跳过"。
            if self.service.topology.is_fact_node(target):
                skipped.append(target)
                continue
            # 先判存在，再撤销。这里刻意**不用** ``except KeyError`` 吞掉"找不到"：
            # 那样会把 forget 内部真正的 KeyError 也伪装成"对象已经不在了"，
            # 而遗忘确实发生了却报告成空列表 —— 一个曾经很难查的假成功。
            relation = self.service.topology.get_relation(target)
            exists = relation is not None or self.service.topology.find_record(target) is not None
            if not exists:
                continue
            try:
                if relation is not None:
                    self.service.forget_role_relation(role_uuid, target)
                else:
                    self.service.forget_role_node(role_uuid, target)
            except Exception as exc:  # noqa: BLE001 - 一个对象撤不掉不能瞒着用户
                errors.append({"object_id": target, "error": f"{type(exc).__name__}: {exc}"})
                continue
            forgotten.append(target)
        self._update(
            job, log=True, rolled_back=True, rolled_back_at=utc_timestamp(),
            status="rolled_back" if not errors else "rollback_partial",
            rollback={"forgotten": forgotten, "skipped_fact_nodes": skipped, "errors": errors},
            message=(
                f"「{job['file']['name']}」已撤销：这个角色不再记得它写入的 {len(forgotten)} 个对象"
                if not errors else
                f"「{job['file']['name']}」只撤销了一部分，{len(errors)} 个对象没撤掉（见 rollback.errors）"
            ),
        )
        return self.public(job)

    # ------------------------------------------------------------------ 公开视图

    @staticmethod
    def public(job: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(job)
        result.pop("published_objects", None)
        result["spec"] = {
            k: v for k, v in (result.get("spec") or {}).items()
            if k in {"display_name", "source_role_name", "source_user_name", "user_appellation"}
        }
        return result
