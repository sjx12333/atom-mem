"""补充世界知识的路由（独立模块，便于和别的角色管理改动分开合并）。

路由挂在 ``create_router`` 已经建好的 ``/graphiti`` 前缀 router 上，而不是自己再建一个
APIRouter：前缀、标签和 OpenAPI 分组都保持一处定义。鉴权复用
``api.management_auth``，与角色管理路由同一份语义。
"""

from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

try:
    from .management_auth import management_access_dependency, record_management_failure
except ImportError:  # 直接以 ``api`` 为顶层包导入时
    from api.management_auth import management_access_dependency, record_management_failure


#: 一次最多几份 txt。没有技术上的硬限制，但一次几百份会让进度页和撤销都变得没法看；
#: 需要更多就分批导入（重复导入同一份会被识别出来）。
MAX_FILES_PER_REQUEST = 50


def create_knowledge_import_routes(router: APIRouter, service: Any, root: Any, configured_token: Optional[str]) -> None:
    """在既有 router 上注册补充知识的四个路由。"""
    require_management_access = management_access_dependency(root, configured_token)

    def coordinator():
        return getattr(service, "role_knowledge_imports", None)

    @router.post("/knowledge-imports", dependencies=[Depends(require_management_access)])
    async def start_knowledge_import(
        request: Request,
        role_uuid: str = Form(...),
        files: List[UploadFile] = File(...),
        user_appellation: Optional[str] = Form(None),
        extract_memories: bool = Form(True),
    ):
        """给一个已有的角色补充若干份 txt 里的世界知识。

        每份文件是一个独立子任务：独立分块、独立记账、独立撤销。记忆批默认开启，
        可以整批关掉（``extract_memories=false``）—— 纯设定集里没有这个角色的经历。
        """
        imports = coordinator()
        if imports is None:
            record_management_failure(root, request, 503, "knowledge import is not available")
            raise HTTPException(status_code=503, detail="knowledge import is not available")
        if not files:
            record_management_failure(root, request, 400, "no files supplied")
            raise HTTPException(status_code=400, detail="至少要选择一份 txt")
        if len(files) > MAX_FILES_PER_REQUEST:
            record_management_failure(
                root, request, 400, f"too many files ({len(files)} > {MAX_FILES_PER_REQUEST})"
            )
            raise HTTPException(
                status_code=400,
                detail=f"一次最多 {MAX_FILES_PER_REQUEST} 份 txt，请分批导入",
            )
        documents = []
        for upload in files:
            name = str(upload.filename or "").strip() or "未命名.txt"
            if not name.lower().endswith(".txt"):
                record_management_failure(root, request, 400, f"not a txt: {name}")
                raise HTTPException(status_code=400, detail=f"「{name}」不是 .txt 文件")
            try:
                content = (await upload.read()).decode("utf-8")
            except UnicodeDecodeError as exc:
                record_management_failure(root, request, 400, f"not utf-8: {name}")
                raise HTTPException(status_code=400, detail=f"「{name}」不是 UTF-8 文本") from exc
            documents.append({"name": name, "content": content})
        try:
            return imports.start(
                role_uuid=role_uuid,
                files=documents,
                user_appellation=str(user_appellation or ""),
                extract_memories=bool(extract_memories),
            )
        except ValueError as exc:
            # 角色不存在、没有模型配置、身份映射缺失、文件为空：都是用户可修的问题，
            # 不是 500。与角色创建路由同一套错误分类。
            record_management_failure(root, request, 400, exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/knowledge-imports", dependencies=[Depends(require_management_access)])
    async def list_knowledge_imports(
        role_uuid: Optional[str] = None, group_id: Optional[str] = None
    ):
        imports = coordinator()
        if imports is None:
            return {"jobs": []}
        return {"jobs": imports.list_jobs(role_uuid=role_uuid, group_id=group_id)}

    @router.get("/knowledge-imports/{job_id}", dependencies=[Depends(require_management_access)])
    async def knowledge_import_status(job_id: str):
        imports = coordinator()
        if imports is None:
            raise HTTPException(status_code=404, detail="knowledge import is not available")
        try:
            return imports.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.post(
        "/knowledge-imports/{job_id}/rollback", dependencies=[Depends(require_management_access)]
    )
    async def rollback_knowledge_import(request: Request, job_id: str):
        """撤销这一次导入：让这个角色不再记得它写入的对象。

        只撤销这个角色对这批对象的访问，**不删除共享图谱对象** —— 别的角色可能
        也认识同一个实体。要删掉共享对象，用图谱管理页的显式删除。
        """
        imports = coordinator()
        if imports is None:
            record_management_failure(root, request, 503, "knowledge import is not available")
            raise HTTPException(status_code=503, detail="knowledge import is not available")
        try:
            return {"job": imports.rollback(job_id)}
        except KeyError as exc:
            record_management_failure(root, request, 404, exc)
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            record_management_failure(root, request, 409, exc)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
