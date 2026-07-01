import json
from json import JSONDecodeError
from typing import Any, Callable

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class MCPClient:
    def __init__(self, session: ClientSession) -> None:
        self.session = session

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        result = await self.session.call_tool(name, arguments=args)
        text = result.content[0].text

        try:
            return json.loads(text)
        except JSONDecodeError:
            return {"raw_text": text}


async def connect_and_run(url: str, token: str, callback: Callable) -> Any:
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(headers=headers, timeout=120) as http_client:
        async with streamable_http_client(url, http_client=http_client) as streams:
            read_stream, write_stream, _ = streams

            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await callback(MCPClient(session))
