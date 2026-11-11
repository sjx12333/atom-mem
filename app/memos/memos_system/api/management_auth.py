"""管理访问鉴权与失败落盘：路由模块共用的两个依赖。

原先这两个函数是 ``graphiti_routes.create_router`` 里的闭包，所以任何**新**的路由
模块（例如补充世界知识）要么把路由塞进那个已经很大的文件，要么自己抄一份鉴权。
两台机器同时改角色管理时，第一种做法是冲突热点，第二种会让"被拒的管理请求"在
诊断日志里时有时无。抽到这里之后，两个模块共用同一份语义。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import HTTPException, Request

# 这个模块被两种方式导入：作为 ``memos_system.api`` 的子模块（相对导入），以及
# 通过 ``graphiti_routes`` 的回退分支作为顶层 ``api`` 模块（绝对导入）。后者要求
# ``memos_system`` 在 ``sys.path`` 上 —— 而 ``4.MEMOS-API.bat`` 用的是
# ``python -E -s -m memos_system.api.graphiti_server``，此时它**不在**。
# 缺这一手，导入会掉进 ``graphiti_routes`` 的 ``except ImportError`` 回退，然后在
# 这里以 ``No module named 'graphiti_memory'`` 失败，整个记忆网关起不来。
_MEMOS_SYSTEM_ROOT = Path(__file__).resolve().parents[1]
if str(_MEMOS_SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEMOS_SYSTEM_ROOT))

try:
    from ..graphiti_memory.api_management import ManagementAccessError
except ImportError:  # 直接以 ``api`` 为顶层包导入时
    from graphiti_memory.api_management import ManagementAccessError


def record_management_failure(root: Path | str, request: Request, status: int, detail: Any) -> None:
    """Append one rejected management request to ``logs/management_failures.jsonl``.

    A management call the server refuses used to leave no trace anywhere: the
    desktop shows a four-second toast that a modal dialog covers, the uvicorn console
    has scrolled away, and nothing on disk records that the call even arrived.
    """
    try:
        path = Path(root) / "logs" / "management_failures.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "method": request.method,
            "path": request.url.path,
            "status": int(status),
            "detail": str(detail)[:500],
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        # Diagnostics must never turn a rejected request into a server error.
        pass


def management_access_dependency(
    root: Path | str, configured_token: Optional[str]
) -> Callable[[Request], None]:
    """Build the ``require_management_access`` dependency for one router."""

    def require_management_access(request: Request) -> None:
        host = request.client.host if request.client else None
        try:
            ManagementAccessError.validate(
                host,
                request.headers.get("x-graphiti-management-session"),
                configured_token,
                origin=request.headers.get("origin"),
            )
        except ManagementAccessError as exc:
            record_management_failure(root, request, 401, str(exc))
            raise HTTPException(status_code=401, detail=str(exc)) from exc

    return require_management_access
