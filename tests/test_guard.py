"""The cross-site guard in front of every writing route (registry rule 11).

    python3 tests/test_guard.py

Three layers. The decorator on a trivial handler: what it refuses (a simple
`text/plain` POST, a JSON POST from another origin) and what it passes (JSON
from this origin, JSON with no Origin at all, a multipart upload wearing the
pack's header). Then the same through a real aiohttp server, since the fake
request is only as honest as its headers. Then the source: every
`routes.post(` in the modules `__init__.py` registers is followed by
`@same_origin` on the next line, so a route added later cannot go out
unguarded.
"""

import asyncio
import json
import os
import re

import layout
from aiohttp import web
from harness import check, passed

layout.stub_server()
guard = layout.load("guard").guard
same_origin, REQUEST_HEADER = guard.same_origin, guard.REQUEST_HEADER

ran = []


@same_origin
async def handler(request):
    ran.append(request)
    return web.json_response({"ok": True})


class Request:
    def __init__(self, **headers):
        self.headers = {k.replace("_", "-"): v for k, v in headers.items()}


def post(**headers):
    ran.clear()
    response = asyncio.run(handler(Request(**headers)))
    return response.status, json.loads(response.text), len(ran)


HOST = "127.0.0.1:8188"
JSON = "application/json"

check("a text/plain POST is refused before the handler runs",
      post(Host=HOST, Content_Type="text/plain"), (415, {"error": f"send {JSON}, or the {REQUEST_HEADER} header"}, 0))
check("so is a form POST", post(Host=HOST, Content_Type="application/x-www-form-urlencoded")[0], 415)
check("and a POST with no Content-Type at all", post(Host=HOST)[0], 415)
check("a JSON POST from another origin is refused",
      post(Host=HOST, Content_Type=JSON, Origin="http://evil.example"),
      (403, {"error": "that request came from another site"}, 0))
check("a same-origin JSON POST runs the handler",
      post(Host=HOST, Content_Type=JSON, Origin="http://127.0.0.1:8188"), (200, {"ok": True}, 1))
check("a JSON POST with no Origin runs the handler (curl, the desktop app)",
      post(Host=HOST, Content_Type=JSON), (200, {"ok": True}, 1))
check("the charset does not matter", post(Host=HOST, Content_Type="application/json; charset=utf-8")[0], 200)
check("a multipart POST wearing the pack's header passes",
      post(**{"Host": HOST, "Content-Type": "multipart/form-data; boundary=x", REQUEST_HEADER: "1"})[0], 200)
check("a multipart POST without it does not",
      post(Host=HOST, Content_Type="multipart/form-data; boundary=x")[0], 415)

# Origin against Host: the port is the whole difference between the tab that
# owns this server and one served from beside it.
check("another port on the same host is another site",
      post(Host=HOST, Content_Type=JSON, Origin="http://127.0.0.1:5173")[0], 403)
check("a Host with no port means the scheme's port",
      post(Host="comfy.lan", Content_Type=JSON, Origin="http://comfy.lan")[0], 200)
check("https behind a proxy: the Host is bare, the Origin says 443",
      post(Host="comfy.lan", Content_Type=JSON, Origin="https://comfy.lan:443")[0], 200)
check("an opaque `null` origin is refused", post(Host=HOST, Content_Type=JSON, Origin="null")[0], 403)
check("so is an origin that is not a URL", post(Host=HOST, Content_Type=JSON, Origin="not a url")[0], 403)
check("and a host that is not one", post(Host="[::1", Content_Type=JSON, Origin="http://[::1")[0], 403)
check("the original handler is reachable for a caller that has its own reason",
      handler.__wrapped__.__name__, "handler")


# ---- through a real server -------------------------------------------------------

try:
    from aiohttp.test_utils import TestClient, TestServer
except ImportError:   # the venv without test utils: the handler-level checks stand
    TestClient = None

if TestClient is not None:
    async def live():
        app = web.Application()
        app.router.add_post("/write", handler)
        async with TestClient(TestServer(app)) as client:
            origin = f"http://{client.host}:{client.port}"
            simple = await client.post("/write", data="{}", headers={"Content-Type": "text/plain"})
            check("live: a simple-request body is refused", simple.status, 415)
            cross = await client.post("/write", json={}, headers={"Origin": "http://evil.example"})
            check("live: a cross-origin JSON POST is refused", cross.status, 403)
            same = await client.post("/write", json={}, headers={"Origin": origin})
            check("live: a same-origin JSON POST lands", same.status, 200)
            bare = await client.post("/write", json={})
            check("live: no Origin lands", bare.status, 200)
            # A cross-site page could only reach the route through a preflight,
            # which nothing here answers — the OPTIONS falls through to 405.
            flight = await client.options("/write", headers={
                "Origin": "http://evil.example", "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type"})
            check("live: the preflight a JSON POST needs is not answered", flight.status, 405)
            check("live: and grants nothing", "Access-Control-Allow-Origin" in flight.headers, False)
    asyncio.run(live())


# ---- every route, in the source --------------------------------------------------

ROOT = os.path.dirname(layout.PY_ROOT)
with open(os.path.join(ROOT, "__init__.py")) as f:
    init = f.read()
modules = []
for package, name in re.findall(r"^from \.creator([\w.]*) import (\w+)", init, re.MULTILINE):
    inside = os.path.join(layout.PY_ROOT, *package.strip(".").split(".")) if package else layout.PY_ROOT
    path = os.path.join(inside, name) + ".py"
    # `from .creator.creator_node import comfy_entrypoint` names a symbol; the
    # module it came from is the one to read.
    modules.append(path if os.path.exists(path) else inside + ".py")
unguarded, guarded = [], 0
for path in modules:
    with open(path) as f:
        lines = f.read().split("\n")
    for i, line in enumerate(lines):
        if "routes.post(" not in line:
            continue
        if lines[i + 1].strip() == "@same_origin":
            guarded += 1
        else:
            unguarded.append(f"{os.path.relpath(path, ROOT)}:{i + 1}")
check("the registered modules were found", len(modules) > 10, True)
check("every POST route is guarded on the line after its decorator", unguarded, [])
check("and the sweep saw them", guarded > 0, True)

passed("all guard tests passed")
