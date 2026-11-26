"""Minimal async Supabase gateway (PostgREST + Storage + Auth admin) built on httpx.

Every data call is made **with the end user's JWT**, so Postgres Row Level Security
and Storage policies are enforced by Supabase itself — the API cannot read or
write another user's rows even if application code had a bug. The service-role
key is used only for `delete_auth_user` / account deletion.
"""
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx

log = logging.getLogger("wastewise.supabase")

# Waste photos are private user data. Shared caches (Supabase's CDN) must not keep a copy,
# otherwise a signed URL can go on serving an image for up to an hour after the user deleted it
# (observed in the production E2E run 45777888257 with the old max-age=3600).
PRIVATE_IMAGE_CACHE_CONTROL = "private, no-store, max-age=0"


class SupabaseError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


class SupabaseGateway:
    def __init__(self, url: str, anon_key: str, service_key: str, http: httpx.AsyncClient):
        self.url = url
        self.anon = anon_key
        self.service = service_key
        self.http = http

    # ------------------------------------------------------------------ helpers
    def _h(self, token: str, extra: dict | None = None) -> dict:
        h = {"apikey": self.anon, "Authorization": f"Bearer {token}"}
        if extra:
            h.update(extra)
        return h

    @staticmethod
    def _raise(r: httpx.Response) -> None:
        if r.status_code >= 400:
            try:
                body = r.json()
                msg = body.get("message") or body.get("error") or body.get("msg") or r.text
            except Exception:
                msg = r.text
            raise SupabaseError(r.status_code, str(msg)[:300])

    # ------------------------------------------------------------------ PostgREST
    async def select(self, token: str, table: str, params: dict[str, str], *, count: bool = False,
                     range_: tuple[int, int] | None = None) -> tuple[list[dict], int | None]:
        extra = {}
        if count:
            extra["Prefer"] = "count=exact"
        if range_:
            extra["Range-Unit"] = "items"
            extra["Range"] = f"{range_[0]}-{range_[1]}"
        r = await self.http.get(f"{self.url}/rest/v1/{table}", params=params, headers=self._h(token, extra))
        if r.status_code == 416:  # range past the end
            total = int(r.headers.get("content-range", "*/0").split("/")[-1] or 0)
            return [], total
        self._raise(r)
        total = None
        if count and "content-range" in r.headers:
            tail = r.headers["content-range"].split("/")[-1]
            total = int(tail) if tail.isdigit() else None
        return r.json(), total

    async def count(self, token: str, table: str, params: dict[str, str]) -> int:
        r = await self.http.head(f"{self.url}/rest/v1/{table}", params=params,
                                 headers=self._h(token, {"Prefer": "count=exact"}))
        self._raise(r)
        tail = r.headers.get("content-range", "*/0").split("/")[-1]
        return int(tail) if tail.isdigit() else 0

    async def insert(self, token: str, table: str, row: dict) -> dict:
        r = await self.http.post(f"{self.url}/rest/v1/{table}", json=row,
                                 headers=self._h(token, {"Prefer": "return=representation"}))
        self._raise(r)
        data = r.json()
        return data[0] if isinstance(data, list) else data

    async def update(self, token: str, table: str, params: dict[str, str], patch: dict) -> list[dict]:
        r = await self.http.patch(f"{self.url}/rest/v1/{table}", params=params, json=patch,
                                  headers=self._h(token, {"Prefer": "return=representation"}))
        self._raise(r)
        return r.json()

    async def delete(self, token: str, table: str, params: dict[str, str]) -> list[dict]:
        r = await self.http.delete(f"{self.url}/rest/v1/{table}", params=params,
                                   headers=self._h(token, {"Prefer": "return=representation"}))
        self._raise(r)
        return r.json()

    # ------------------------------------------------------------------ Storage
    async def upload(self, token: str, bucket: str, path: str, data: bytes, content_type: str,
                     upsert: bool = False) -> None:
        r = await self.http.post(f"{self.url}/storage/v1/object/{bucket}/{quote(path)}", content=data,
                                 headers=self._h(token, {"Content-Type": content_type,
                                                         "x-upsert": "true" if upsert else "false",
                                                         "cache-control": PRIVATE_IMAGE_CACHE_CONTROL}))
        self._raise(r)

    async def download(self, token: str, bucket: str, path: str) -> bytes:
        r = await self.http.get(f"{self.url}/storage/v1/object/authenticated/{bucket}/{quote(path)}",
                                headers=self._h(token))
        self._raise(r)
        return r.content

    async def remove(self, token: str, bucket: str, paths: list[str]) -> None:
        if not paths:
            return
        r = await self.http.request("DELETE", f"{self.url}/storage/v1/object/{bucket}",
                                    json={"prefixes": paths}, headers=self._h(token))
        self._raise(r)

    async def sign_urls(self, token: str, bucket: str, paths: list[str], expires_in: int) -> dict[str, str]:
        paths = [p for p in paths if p]
        if not paths:
            return {}
        r = await self.http.post(f"{self.url}/storage/v1/object/sign/{bucket}",
                                 json={"expiresIn": expires_in, "paths": paths}, headers=self._h(token))
        self._raise(r)
        out = {}
        for item in r.json():
            signed = item.get("signedURL") or item.get("signedUrl")
            if signed and not item.get("error"):
                out[item["path"]] = f"{self.url}/storage/v1{signed}" if signed.startswith("/") else signed
        return out

    async def list_objects(self, token: str, bucket: str, prefix: str, limit: int = 1000) -> list[dict]:
        r = await self.http.post(f"{self.url}/storage/v1/object/list/{bucket}",
                                 json={"prefix": prefix, "limit": limit, "offset": 0},
                                 headers=self._h(token))
        self._raise(r)
        return r.json()

    # ------------------------------------------------------------------ Auth admin (service role)
    async def delete_auth_user(self, user_id: str) -> None:
        if not self.service:
            raise SupabaseError(503, "account deletion is not configured on the server")
        r = await self.http.delete(f"{self.url}/auth/v1/admin/users/{user_id}",
                                   headers={"apikey": self.service, "Authorization": f"Bearer {self.service}"})
        self._raise(r)

    async def ping(self) -> bool:
        try:
            r = await self.http.get(f"{self.url}/auth/v1/health", headers={"apikey": self.anon}, timeout=4)
            return r.status_code == 200
        except Exception:
            return False


def pg_eq(value: Any) -> str:
    return f"eq.{value}"
