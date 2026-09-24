"""An uncut sheet is answered inside the request; a cut one takes its turn (#89).

    python3 tests/test_plate_route.py

`/continuity/plate` used to queue every sheet behind whatever was rendering.
A sheet with nothing cut out is a resize and a paste — no weights — so making
it wait for a render greyed the picker's Add for the render's length with
nothing saying why, and Cancel was the only way out. The route function is run
here with the queue and the builder replaced: what is checked is which door
each kind of sheet leaves by.
"""

import ast
import asyncio
import json
import logging
from pathlib import Path

from aiohttp import web

import layout
from harness import check, passed

logging.disable(logging.CRITICAL)   # the refused build is logged on purpose

source = Path(layout.PY_ROOT) / "server_routes.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
route = next(node for node in tree.body
             if isinstance(node, ast.AsyncFunctionDef) and node.name == "build_plate")
route.decorator_list = []

built, submitted = [], []


class _Jobs:
    class JobError(ValueError):
        pass

    @staticmethod
    async def submit(kind, body, client_id=None):
        submitted.append((kind, client_id))
        return "prompt-7"


def _plate_job(body):
    built.append(body)
    if body.get("fail"):
        raise ValueError("no such picture")
    return {"path": "_plates/plate-abc.png", "panels": len(body["panels"])}


namespace = {"asyncio": asyncio, "logging": logging, "web": web, "jobs": _Jobs,
             "_plate_panels": lambda body: body["panels"], "_plate_job": _plate_job}
exec(compile(ast.Module(body=[route], type_ignores=[]), str(source), "exec"), namespace)
build_plate = namespace["build_plate"]


class Request:
    headers = {"Content-Type": "application/json"}   # past the guard; test_guard.py holds it

    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


def post(body):
    response = asyncio.run(build_plate(Request(body)))
    return response.status, json.loads(response.text)


status, answer = post({"panels": [{"path": "a.png"}, {"path": "b.png", "cut": False}],
                       "client_id": "tab"})
check("an uncut sheet is answered in the reply", status, 200)
check("with the built plate", answer["result"]["panels"], 2)
check("and nothing on the queue", submitted, [])

status, answer = post({"panels": [{"path": "a.png"}, {"path": "b.png", "cut": True}],
                       "client_id": "tab"})
check("a cut sheet is queued", answer, {"prompt_id": "prompt-7"})
check("under its kind, addressed to the tab", submitted, [("plate", "tab")])
check("and not built inside the request", len(built), 1)

status, answer = post({"panels": [{"path": "gone.png"}], "fail": True})
check("a plate that cannot be built is the user's to act on", status, 400)
check("with the reason", answer["error"], "no such picture")

status, answer = post({"panels": []})
check("no pictures is refused before either door", status, 400)

passed("an uncut sheet does not wait for the render")
