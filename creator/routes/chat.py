"""`/continuity/chat/*`: the two doors the chat room knocks on.

The room is a conversation and a queue, and those are the two things the server
has to hold up. `chat/turn` asks the refiner model what to do about one user
message and answers `{say, action?, raw}`. `chat/render` takes the action it
answered with, turns it into a blob, compiles that blob as a dry run, and puts a
one-node prompt on ComfyUI's queue. Everything between them lives in the
browser: the conversation, the ledger of what has been made, the rail. This
module is stateless on purpose — the spec's §5.5 — so a reload starts fresh and
nothing here has a session to lose.

**The thinking is next door.** `creator/chat.py` owns the action schema, the
parse, the re-ask protocol, the machine card and the two blob patches, and it is
pure. What is here is the joining that cannot be: which files are on this disk,
what this machine last picked, what the compiler says about the result, and the
queue. That is the same line `refine_routes.py` draws against
`families/refine.py`, and it is why the whole protocol can be tested without a
server and tuned on `tools/chat_bench.py` without a browser.

**Two backends, one shape.** The remote refiner is answered inside the request,
exactly as the refine button's is and for the same reason: it spends somebody
else's GPU, so there is nothing for it to queue behind. The in-process one is a
4B model on the card the render wants, so it rides the queue as a `chat` job and
the reply arrives on `executed` under `ui.continuity`. The turn object is the
same either way, and the remote one is wrapped in `{"result": …}` because that
is the envelope `queue.run()` reads: it resolves with `answer.result` where the
reply carries one and waits for `executed` where it does not, so one call site
in the room covers both backends. `/continuity/refine` says it the same way.

**The re-ask is a second call inside the same job**, never a round trip to the
browser. The room asked one question and is owed one answer; a correction the
person watches happen reads as a failure, and on the queued path it would be a
second wait behind whatever is running.

No native tool calls yet. Where a runtime reports tool support the same schema
is the one tool, but the fenced form is the fallback and the default, and it is
what both backends get here — the seam is `_ask` below, which is the one place
a reply is produced.
"""

import asyncio
import json

from aiohttp import web

from server import PromptServer

from .. import chat, jobs, media, models as core_models, refine_local, refine_remote
from .. import refine_routes, refine_skill, server_routes, settings
from ..families import manifest, refine, registry
from ..guard import same_origin


# ---- the machine card -------------------------------------------------------
#
# Cached because it is a join of three payloads — every family's manifest, a walk
# of the model directories, and the settings file — and the room asks for it on
# every turn of a conversation that may run for an hour.

_CARDS = {}
_CARDS_KEEP = 4


def machine_card(still_family, video_family, weights, turbo=False):
    """What this machine can make, as the model reads it. See `chat.machine_card`.

    Keyed on its own inputs rather than invalidated from `settings.save`. A hook
    there would catch the settings page and miss everything else that moves this
    answer — the weights popover patches the same file from the node, and the
    rail will patch it again from the room — whereas the key is exactly the
    facts the card is made of, so anything that changes one of them changes the
    key. `turbo` is one of them: with the switch thrown the render loads the
    Turbo checkpoint instead of the ordinary one, so which files are missing is
    a different answer. The listing is not in the key: a walk of the model
    folders on every turn is the cost this cache exists to avoid, and
    `settings.weights` moving is what a *picked* file changing looks like here.
    """
    key = json.dumps([still_family, video_family, weights, bool(turbo)],
                     sort_keys=True, default=str)
    card = _CARDS.get(key)
    if card is None:
        card = chat.machine_card(still_family, video_family, manifest.catalog(),
                                 core_models.available(), weights, turbo=bool(turbo))
        while len(_CARDS) >= _CARDS_KEEP:
            _CARDS.pop(next(iter(_CARDS)))
        _CARDS[key] = card
    return card


# ---- the rail ---------------------------------------------------------------


def _arch_of(family):
    """The pre-stage pill's name for a family, or None where it makes no stills.

    "minimax" is H3's and is frozen in every saved blob, which is why the pill's
    vocabulary and the registry's ids are not the same words — see
    `registry.STILL_ARCHES`. The rail arrives naming families, because that is
    what the machine card and the settings block are keyed by, so the mapping
    happens once, here, and the pure half is handed the arch.
    """
    for arch, owner in registry.STILL_ARCHES.items():
        if owner == family:
            return arch
    return None


def _rail(raw):
    """The room's rail, filled in and with the still family's arch stamped on.

    Absent families fall back to the registry's own defaults, which is what a
    room with no node under it would be holding, and the same answer the node
    gives a freshly dropped blob.

    Both routes pass it the same object: `chat/render` calls it the rail and
    `chat/turn` sends it inside `settings` beside the backend and the model,
    because a turn has to be told the same things a render does — which
    families, and whether turbo is thrown — before it can say what this machine
    can make. The room reads both families off the nodes on the canvas (the
    piece's family, the pre-stage's arch) and sends them here by name.

    The still side is the *image* families and not every family that makes a
    still. H3's still branch is a video generation with one latent frame decoded
    as a picture, and its pre-stage blob is the Creator's own request under a
    `minimax` block rather than the shared image shape — a second blob for this
    room to patch, for a picture the video family can already be asked for
    directly. So the rail offers the families that draw a picture outright, and
    the room's own pill offers the same set.
    """
    rail = dict(raw or {})
    still = rail.get("still_family")
    if still not in registry.IMAGE_FAMILIES:
        still = registry.STILL_ARCHES[registry.DEFAULT_STILL_ARCH]
    video = rail.get("video_family")
    if video not in registry.video_families():
        video = registry.DEFAULT_VIDEO
    rail["still_family"] = still
    rail["video_family"] = video
    rail["still_arch"] = _arch_of(still)
    return rail


def _base(raw, still):
    """The node's blob and widgets the render is built over -> `(piece, widgets)`.

    `{piece, widgets}` as the room sends it: the pre-stage's `prestage_data` or
    the Creator's `creator_data`, parsed, and the node's sampler widgets by
    name — the seed above all, since the seed is the one number on the row that
    never moved into the blob. A base of the wrong kind is refused rather than
    rendered: a pre-stage blob has an `arch` and a piece has a `family`, and a
    still built over a clip's piece would carry a strip into `compile_prestage`.
    """
    raw = raw if isinstance(raw, dict) else {}
    piece = raw.get("piece")
    if isinstance(piece, str):
        try:
            piece = json.loads(piece)
        except ValueError:
            piece = None
    if not isinstance(piece, dict):
        raise chat.ActionError("the room sent no node to render with.")
    if still and "arch" not in piece:
        raise chat.ActionError("the still's base is not a pre-stage blob.")
    if not still and "arch" in piece:
        raise chat.ActionError("the clip's base is not a piece.")
    widgets = raw.get("widgets")
    return piece, dict(widgets) if isinstance(widgets, dict) else {}


def _families_of(piece, still):
    """The family a base names, as the rail spells it — so the render is built
    for the node that was sent and not for whatever the rail last said.

    A pre-stage names an arch, and the room can only draw a picture through
    an image arch: H3's still branch is a video generation under its own blob
    (`_rail` says why), so a pre-stage left on it is refused here, in the
    room's words, rather than built into the wrong compiler.
    """
    if not still:
        return {"video_family": piece.get("family")}
    family = registry.STILL_ARCHES.get(piece.get("arch"))
    if family not in registry.IMAGE_FAMILIES:
        raise chat.ActionError(
            "the pre-stage is on MiniMax H3, whose still the room cannot ask "
            "for. Pick an image model on it and ask again.")
    return {"still_family": family}


# ---- the weights ------------------------------------------------------------


def _picked(family, stored, available=None, own=None):
    """The files `family` renders with, as `{slot: filename}`.

    The node's own block (`own`) over this machine's remembered picks over the
    folder listing's own guess — the node's rule, `adoptWeights`, read back: a
    piece says what it rendered on, and memory only fills what it left empty.
    The remembered half is the
    same join `chat.setup_report` shows the room, so a family the room says is
    ready is a family this route renders: a machine that was never asked to
    pick (the family pill moved without a first run, a fresh install with one
    file per slot) still renders on the files it has. Filtered to the slots the
    family actually declares. `settings.clean_weights` stores a block as
    written — a slot id it has never heard of is kept, because what a slot
    *means* is the family's — so a block left behind by another family, or by
    a version of this one with a slot since renamed, would otherwise ride into
    the blob and read as nothing to every reader downstream.
    """
    ids = {slot["id"] for slot in family.get("weights") or []}

    def files(block):
        return {name: value for name, value in (block or {}).items()
                if name in ids and isinstance(value, str) and value.strip()}

    guessed = chat.guess_weights(family, available) if available is not None else {}
    return {**guessed, **files((stored or {}).get(family["id"])), **files(own)}


def _not_ready(family, picked, available, turbo=False):
    """The sentence a family with a required file missing gets, or None.

    The same reading the machine card takes, so what the model was told and what
    the render refuses cannot disagree — the card says a family is not ready and
    this is the refusal if the model asks for it anyway. `turbo` is the rail's
    switch, which moves the answer: with it thrown the Turbo checkpoint is the
    file this render loads, and without this the dry run would pass and
    `render_image.check` would refuse on the queue instead.
    """
    gone = chat.missing_weights(family, picked, available, turbo=turbo)
    if not gone:
        return None
    return (f"{family['label']} cannot render yet: no file is picked for "
            f"{chat.listed(gone)}. Choose {'it' if len(gone) == 1 else 'them'} "
            f"in the weights control and ask again.")


# ---- building the render ----------------------------------------------------


def _node_widgets(node, family, own):
    """Every widget a node's schema declares, as the node on the canvas has them.

    A prompt built by hand carries the whole input list: ComfyUI validates a
    prompt against the schema, and a widget left off is a render that will not
    queue. The values are the schema's own defaults read off the schema — the
    node is the only honest source for what it declares — with the family's
    manifest laid over them, because a static schema can only wear one family's
    numbers and the manifest is where a family says what row it samples on,
    and the node's own widget values (`own`, as the room read them) over both.
    `sampling.py` argues the blob-over-widget split at length; this is the
    headless end of it, and the blob wins there too, so what the row is set
    to on the node is what samples here.

    The seed is among them and is always a widget. `control_after_generate` is
    the frontend's own linked control and there is nothing in a blob for it to
    be, which is why the seed is the one number on this row that never moved —
    and why the room rolls it on the node after a queue, the way the frontend
    does, rather than asking this route to.
    """
    values = {}
    for spec in node.define_schema().inputs:
        name = getattr(spec, "id", None) or getattr(spec, "name", None)
        default = getattr(spec, "default", None)
        if name and default is not None:
            values[name] = default
    for widget in family.get("widgets") or []:
        if widget.get("id") in values and widget.get("default") is not None:
            values[widget["id"]] = widget["default"]
    for name, value in (own or {}).items():
        if name in values and isinstance(value, (str, int, float, bool)):
            values[name] = value
    return values


def _nodes():
    """The two node classes a chat render is built around.

    Imported here rather than at module scope: both pull in ComfyUI, and route
    modules are imported while the server is still registering its routes.
    """
    from .. import creator_node, prestage

    return {"MiniMaxH3PreStage": prestage.MiniMaxH3PreStage,
            "MiniMaxH3Creator": creator_node.MiniMaxH3Creator}


def _build(action, ledger, rail, stored, base, strip=None, cast=None):
    """The blocking half of a render: patch, fill, dry run, build the prompt.

    -> `{"piece": blob, "prompt": one-node prompt}`, or `{"problem": sentence}`.
    A problem is the assistant's next line, verbatim, and is not an error: a
    duration off the frame grid and a checkpoint nobody picked are both ordinary
    things for a model to have asked for, and relaying the compiler's own
    sentence is what lets it say so and try again.

    `base` is `(piece, widgets)` — the node on the canvas, as the room read it
    and `_base` parsed it, with the rail already naming its family. `strip`
    is the shots the room has joined so far, each with its take — the room's
    state, sent the way the ledger is. `cast` is the piece's subjects as the
    room read them, for a still: a picture has no cast, and a member cited
    in one becomes their picture (`chat.still_piece`). The
    render is that node asked for this prompt in this shape: its stack,
    its turbo switch, its sampler row and its weights are the ones on the
    canvas, and nothing here re-derives any of them. The room's gear is the
    node's own row over the same blob, so there is no second place they could
    have been set.

    The patched blob goes back with the prompt id so the browser can hold it
    without patching a second copy of its own. Two implementations of the same
    patch is exactly the drift this pack keeps writing about, and the server has
    to build the blob anyway to queue it.
    """
    still = action["kind"] == chat.KIND_STILL
    piece_in, widgets_in = base
    family = manifest.describe(rail["still_family"] if still else rail["video_family"])
    available = core_models.available()
    own = (piece_in.get("models") or {}).get(rail["still_arch"]) if still \
        else piece_in.get("models")
    picked = _picked(family, stored, available, own=own)

    # Whether the pre-stage's switch loads the Turbo checkpoint decides whether
    # that is a file this render needs.
    problem = _not_ready(family, picked, available,
                         turbo=still and chat.still_turbo_checkpoint(piece_in))
    if problem:
        return {"problem": problem}

    # What the pure half has to know about this family's pictures, which is the
    # catalog's answer and so the route's to look up. Only for a still: on a
    # clip every handle is a reference or the first frame, and the video
    # families read both.
    if still:
        rail = {**rail, "still_pictures": chat.still_pictures(family, manifest.catalog())}
    node_id, field, piece = chat.piece_of(action, ledger, rail, piece_in, strip, cast)

    if still:
        # Per-arch sub-blocks with one shared precision, which is the shape
        # `render_image.ImageWeights.from_blob` reads and the shape the pre-stage
        # writes: flipping the model pill must not forget the other side's files.
        piece["models"] = {**(piece.get("models") or {}),
                           piece["arch"]: picked,
                           "dtype": (piece.get("models") or {}).get("dtype")
                                    or ((stored or {}).get(family["id"]) or {}).get("dtype")
                                    or "default"}
    else:
        # Flat, and the whole block rather than the filenames alone:
        # `models.Weights.from_blob` reads the precision, the route and the
        # pinned devices out of the same dict. The node's block, with what the
        # machine remembers filling its empty rows.
        piece["models"] = {**((stored or {}).get(family["id"]) or {}),
                           **(piece.get("models") or {}), **picked}

    try:
        if still:
            registry.still(piece["arch"]).compile_still(piece, media.image_size)
        else:
            server_routes.compiled_passes(piece)
    except server_routes.COMPILE_FAILURES as refusal:
        return {"problem": str(refusal)}

    node = _nodes()[node_id]
    widgets = _node_widgets(node, family, widgets_in)
    # The blob last: the schema's own default for that widget is in `widgets`
    # too — every input is, which is the point — and it is the one the render is
    # replacing.
    return {"piece": piece, "node": node_id, "field": field,
            "prompt": {"1": {"class_type": node_id,
                             "inputs": {**widgets,
                                        field: json.dumps(piece, indent=2)}}}}


# ---- the Refine switch --------------------------------------------------------


def _has_prompting(family):
    """Whether `family` has a prompt refiner at all.

    Only the video families do: `refine.of` imports `families/<id>/refine.py`,
    and the families that draw a picture outright have none — the Refine button
    is not on the pre-stage either. So the switch is the clip side's, and a
    still goes as the model wrote it, which the rail's hint says.
    """
    try:
        refine.of(family)
    except ModuleNotFoundError as missing:
        # Only the family's own module being absent means "no refiner". A
        # video family's refine module failing on something it imports is a
        # broken install, and a switch that quietly skipped the rewrite over it
        # would be the silent failure the switch was refused for in the first
        # place.
        if (missing.name or "").endswith(f".{family}.refine"):
            return False
        raise
    return True


def _refine(built, block):
    """Put the chat's prompt through the family's own prompting, in place.

    -> `{"model", "problems", "skill"?}` for the card, or `{"problem"}` when the
    compiler will not take the rewrite. The blob in `built` is patched and the
    prompt's widget re-dumped, so what is queued is what was refined.

    The same call the button makes — `refine_routes._run` on a `creator`
    target — with the refiner's own settings, which the room sends as the
    button would: this is the switch the spec's §5.1 describes, a second model
    call before queueing rather than a second copy of the prompting.

    Dry-run again afterwards. The refine route reports a stray handle as a
    problem and goes on, which is right under a panel with a Revert on it; here
    the render is about to go out, and the compiler's own refusal of a rewrite
    is the sentence the room should show instead of a failed queue item.
    """
    piece = built["piece"]
    result = refine_routes._run(chat.refine_request(block, piece))
    problems = chat.refine_into(piece, result, block.get("model"))
    try:
        server_routes.compiled_passes(piece)
    except server_routes.COMPILE_FAILURES as refusal:
        return {"problem": f"The rewrite could not be used: {refusal}"}
    built["prompt"]["1"]["inputs"][built["field"]] = json.dumps(piece, indent=2)
    out = {"model": block.get("model") or "", "problems": problems}
    if result.get("skill"):
        out["skill"] = result["skill"]
    return out


def _enqueue_from_job(prompt, client_id):
    """Put a render on the queue from inside a running job. -> its `prompt_id`.

    The runner is on `prompt_worker`'s thread; `jobs.enqueue` is a coroutine
    that has to run on the server's loop, which is free — the loop is not what
    is executing this node. A render queued this way lands behind the job that
    queued it, which is the order it should run in.
    """
    future = asyncio.run_coroutine_threadsafe(
        jobs.enqueue(prompt, client_id), PromptServer.instance.loop)
    return future.result(timeout=60)


# What a refine can raise that is a sentence for the room rather than a bug:
# the refiner's own refusals, this module's, a file that will not open, and
# whatever the compiler refuses a rewrite for.
_REFINE_FAILURES = (refine.RefineError, chat.ActionError, media.MediaError,
                    *server_routes.COMPILE_FAILURES)


def _render_job(body):
    """A refine and then a render, as one job. See `chat_render`.

    Both halves of the answer are the render's: `{prompt_id, piece, refined}`
    goes on `executed` for the room to hang its card on, exactly as the inline
    reply does, and `{problem}` is the assistant's line when the rewrite cannot
    be rendered.
    """
    try:
        built = body["built"]
        refined = _refine(built, body.get("refine") or {})
        if "problem" in refined:
            return refined
        prompt_id = _enqueue_from_job(built["prompt"], body.get("client_id"))
    except (*_REFINE_FAILURES, jobs.JobError) as problem:
        raise jobs.JobError(str(problem)) from problem
    return {"prompt_id": prompt_id, "piece": built["piece"], "refined": refined}


jobs.register("chat-render", _render_job)


def _no_model(block):
    """The 400 a request with no refiner model gets, or None. Said once."""
    if (block.get("model") or "").strip():
        return None
    if block.get("backend") == "remote":
        return web.json_response({"error":
            "No model chosen. Pick one of the server's models in the "
            "refiner's settings."
        }, status=400)
    return web.json_response({"error":
        "No text encoder chosen. Put a Qwen3-VL or Qwen3.5 text encoder in "
        "models/text_encoders and pick it in the refiner's settings."
    }, status=400)


@PromptServer.instance.routes.post("/continuity/chat/render")
@same_origin
async def chat_render(request):
    """Render what the model asked for.

    -> `{"result": {prompt_id, piece, refined?}}` once the render is on the
    queue, `{"result": {problem}}` when it cannot be, or `{"prompt_id"}` of a job
    that will put the same object on `executed` — the envelope `queue.run()`
    reads, so the room has one call site whichever way the answer comes.

    An ordinary queue item: Cancel reaches it, the progress bar is the real one,
    a render queued from the canvas ahead of it goes first, and what comes out
    is a file in the output folder with its blob in its metadata — citable from
    the picker like anything else this pack makes.

    **The Refine switch.** With it on, a clip's prompt goes through the family's
    own prompting before queueing, exactly as the button does. On the remote
    backend that is a call inside this request, as the refine route's is; on the
    in-process one it is a model on the GPU, so the refine and the render go on
    the queue as one job (`chat-render`) that does the rewrite and then queues
    the render itself. A still is not refined — the families that draw one have
    no prompt refiner, and the rail's hint says so — and goes as written.
    """
    try:
        body = await request.json()
    except ValueError:
        return web.json_response({"error": "the request body was not JSON"}, status=400)

    strip = [c for c in body.get("strip") or [] if isinstance(c, dict)]
    # The piece's shelf and cast, off the blob the room sends beside the
    # base: the handles and the names an action may cite beyond the ledger.
    # For a clip the piece itself is the authority and the base carries the
    # same blob; for a still they are what `still_piece` expands.
    ledger, cast = _with_piece(body)
    try:
        action = chat.validate(body.get("action"), ledger,
                               strip=[c.get(chat.STRIP_KEY) for c in strip],
                               cast=[m.get("name") for m in cast])
        if action["act"] != chat.ACT_RENDER:
            raise chat.ActionError("that action says nothing to render")
        # The node under the room, and the rail naming its family — the render
        # is built for the node that was sent, whatever the rail last said.
        still = action["kind"] == chat.KIND_STILL
        base = _base(body.get("base"), still)
        rail = _rail({**(body.get("rail") or {}), **_families_of(base[0], still)})
    except chat.ActionError as problem:
        return web.json_response({"error": str(problem)}, status=400)

    refining = (bool(rail.get("refine")) and action["kind"] == chat.KIND_VIDEO
                and _has_prompting(rail["video_family"]))
    block = body.get("refine") or {}
    if refining:
        refused = _no_model(block)
        if refused is not None:
            return refused

    # Everything below this line touches disk — the settings file, the model
    # folders, the compiler's size lookups — and this loop is also the prompt
    # queue and the websocket.
    loop = asyncio.get_running_loop()
    try:
        built = await loop.run_in_executor(
            None, lambda: _build(action, ledger, rail,
                                 settings.load().get("weights") or {}, base,
                                 strip, cast))
    except chat.ActionError as problem:
        return web.json_response({"error": str(problem)}, status=400)
    except Exception as problem:  # noqa: BLE001
        return web.json_response({"error": f"{type(problem).__name__}: {problem}"},
                                 status=500)
    if "problem" in built:
        return web.json_response({"result": built})

    client_id = body.get("client_id")
    if refining and block.get("backend") != "remote":
        try:
            prompt_id = await jobs.submit(
                "chat-render", {"built": built, "refine": block, "client_id": client_id},
                client_id)
        except jobs.JobError as problem:
            return web.json_response({"error": str(problem)}, status=500)
        return web.json_response({"prompt_id": prompt_id})

    refined = None
    if refining:
        try:
            refined = await loop.run_in_executor(None, _refine, built, block)
        except _REFINE_FAILURES as problem:
            return web.json_response({"error": str(problem)}, status=400)
        except Exception as problem:  # noqa: BLE001
            return web.json_response({"error": f"{type(problem).__name__}: {problem}"},
                                     status=500)
        if "problem" in refined:
            return web.json_response({"result": refined})

    try:
        prompt_id = await jobs.enqueue(built["prompt"], client_id)
    except jobs.JobError as problem:
        return web.json_response({"error": str(problem)}, status=500)
    result = {"prompt_id": prompt_id, "piece": built["piece"]}
    if refined is not None:
        result["refined"] = refined
    return web.json_response({"result": result})


# ---- the turn ---------------------------------------------------------------


def _skill(block):
    """The user's own instructions for this room, or "". Raises `RefineError`.

    Only the `add` mode. A `.skill` set to *replace* the built-in prompting
    takes the whole system prompt over, and this room's system prompt is the
    contract that turns an answer into a render — replaced, there would be
    nothing to parse and nothing to queue. So it is refused by name rather than
    quietly downgraded to `add`, which would be running a different thing than
    the switch says.
    """
    name = str(block.get("skill") or "").strip()
    if not name:
        return ""
    skill = refine_skill.load(name)
    if (block.get("skill_mode") or skill["mode"]) != refine_skill.ADD:
        raise refine.RefineError(
            f"'{name}' is set to replace the built-in prompting, and the chat "
            f"cannot run it that way — its reply contract is what turns an "
            f"answer into a render. Set it to add to the built-in prompting, "
            f"or pick no skill here."
        )
    return refine_skill.instructions(skill)


def _ask(block, system, message):
    """One generation on whichever backend the room is pointed at.

    The one place a reply is produced, which is where a native tool call will
    land when the runtime reports it: the same schema as the one tool, with
    `parallel_tool_calls` off, and this fenced form as the fallback. Until then
    both backends get the fence.

    `prefill=""` rather than the harness's opening brace. The refiner's replies
    are a bare JSON object and beginning the assistant's turn inside it is what
    stops a small model answering the request instead of expanding it; here the
    reply is a line of plan *and then* an object, so a prefilled `{` would cut
    the plan off before it was written. `json_object` finds the brace either way.
    """
    ask, _look = refine_routes._backend(block)
    return ask(
        block.get("model") or "",
        system, message, [],
        # Choosing an action is a decision, not ideation: the default leans cold
        # so that "another one, bluer" stays a render rather than becoming a
        # conversation about one. The dial is still the user's.
        temperature=block.get("temperature", 0.3),
        seed=block.get("seed", -1),
        max_tokens=block.get("max_tokens"),
        prefill="",
    )


def _with_piece(body):
    """The ledger with the piece's shelf on it, and the piece's cast.

    `body["piece"]` is the video piece under the room — the node's blob or
    the pinned copy — as a dict or as the widget's JSON string. Both routes
    read it the same way, so what the model was told a handle means and
    what the render attaches under it cannot disagree.
    """
    piece = body.get("piece")
    if isinstance(piece, str):
        try:
            piece = json.loads(piece)
        except ValueError:
            piece = None
    piece = piece if isinstance(piece, dict) else {}
    ledger = list(body.get("ledger") or []) + chat.shelf_entries(piece)
    return ledger, chat.cast_entries(piece)


def _last_user(messages):
    """The message this turn is answering — the newest user turn, or ""."""
    for message in reversed(messages or []):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("text") or "")
    return ""


def _run(body):
    """The blocking half of a turn: build the context, ask, judge, maybe re-ask.

    -> `{say, action?, raw, reask?}`. `raw` is the reply the answer came out of
    and `reask` is the sentence that was quoted when there were two, so the room
    can show what happened rather than only what survived.
    """
    block = body.get("settings") or {}
    messages = body.get("messages") or []
    rail = _rail(block)
    # The room's strip and the piece's shelf and cast, as the model is told
    # them: the shots joined so far as `{handle, seconds}`, the shelf's files
    # as ledger lines beside the room's own, and the members off the node's
    # blob. The validator is given the same handles and names, so an "after"
    # or an "@anna" is judged against exactly what was shown.
    strip = [c for c in body.get("strip") or [] if isinstance(c, dict) and c.get("handle")]
    ledger, cast = _with_piece(body)
    on_strip = [c["handle"] for c in strip]
    names = [m["name"] for m in cast]

    # Whether the pre-stage's switch loads the Turbo checkpoint — read off its
    # blob by the room and sent as one flag, since the card only needs to know
    # which files a still would load.
    card = machine_card(rail["still_family"], rail["video_family"],
                        settings.load().get("weights") or {},
                        turbo=bool(block.get("still_turbo_checkpoint")))
    system = chat.system_prompt(_skill(block))
    message = chat.context(messages, ledger, card, strip=strip, cast=cast)
    asked = _last_user(messages)

    raw = _ask(block, system, message)
    verdict = chat.judge(raw, ledger, asked, strip=on_strip, cast=names)
    quoted = None
    if verdict["act"] == "reask":
        quoted = verdict["sentence"]
        raw = _ask(block, system, chat.reask(message, raw, quoted))
        verdict = chat.judge(raw, ledger, asked, second=True, strip=on_strip, cast=names)

    out = {"say": verdict.get("say") or "", "raw": raw}
    if quoted:
        out["reask"] = quoted
    if verdict["act"] == chat.ACT_RENDER:
        out["action"] = verdict["action"]
    return out


def _run_job(body):
    """One turn, on the queue. See `creator/jobs.py`.

    The pack's own three become a message about the request, the way the refine
    job's do: the room puts the sentence in the conversation either way, and
    anything else is a bug reported as one.
    """
    try:
        return _run(body)
    except (refine.RefineError, chat.ActionError) as problem:
        raise jobs.JobError(str(problem)) from problem


jobs.register("chat", _run_job)


@PromptServer.instance.routes.post("/continuity/chat/turn")
@same_origin
async def chat_turn(request):
    """Ask the model what to do about one message. -> the turn, or a `prompt_id`.

    Stateless: the browser sends the trimmed history, the ledger and the rail
    every time, and this builds one turn out of them. The remote backend answers
    inside the request, as `{"result": turn}`; the in-process one is GPU work
    and rides the queue, answering `{"prompt_id": id}` and putting the same turn
    on `executed`. A local chat waits behind a running render — that is the cost
    of one GPU, and the room shows the queue position rather than pretending
    otherwise.
    """
    try:
        body = await request.json()
    except ValueError:
        return web.json_response({"error": "the request body was not JSON"}, status=400)

    block = body.get("settings") or {}
    remote = block.get("backend") == "remote"
    refused = _no_model(block)
    if refused is not None:
        return refused

    if remote:
        # Off the event loop even so: the call to the server blocks, and this
        # loop is also the prompt queue and the websocket.
        loop = asyncio.get_running_loop()
        try:
            turn = await loop.run_in_executor(None, _run, body)
            # Under `result`, which is the envelope `queue.run()` reads for a
            # reply that did not need the queue. The queued path puts the same
            # object on `executed` instead, so the room has one call site and
            # one shape whichever backend it is pointed at.
            return web.json_response({"result": turn})
        except (refine.RefineError, chat.ActionError) as problem:
            return web.json_response({"error": str(problem)}, status=400)
        except Exception as problem:  # noqa: BLE001
            return web.json_response({"error": f"{type(problem).__name__}: {problem}"},
                                     status=500)

    try:
        prompt_id = await jobs.submit("chat", body, body.get("client_id"))
    except jobs.JobError as problem:
        return web.json_response({"error": str(problem)}, status=500)
    return web.json_response({"prompt_id": prompt_id})


# ---- the first run ----------------------------------------------------------


def _local_refiner(names):
    """The text encoder the room would think with, out of the folder listing.

    The kinds ComfyUI can generate from are named in `refine_local.SUPPORTED`,
    smallest first; a file whose name says one of them is the answer, in that
    order, because the room's turn is a queue slot on the card the
    render wants. Nothing recognised means the person is offered the list.
    """
    for wanted in refine_local.SUPPORTED:
        needle = wanted.replace("_", "")
        for name in names:
            if needle in name.lower().replace("_", "").replace("-", ""):
                return name
    return ""


def _setup():
    """Everything the room's first three questions are asked against.

    One walk of the model folders and one knock on each loopback port, joined
    with what this machine has already picked — the same three payloads the
    machine card is a join of, read once and handed to the browser whole so
    the questions can be answered without a request per chip.
    """
    available = core_models.available()
    local = refine_local.list_models()
    stored = refine_remote.status()
    servers = refine_remote.probe_local()
    # The server somebody already set up, listed by the same shape as the
    # loopback ones so the room can offer it as a chip — but only when it
    # answers, and never a second time when it *is* one of the loopback ones.
    if stored.get("url") and stored["url"] not in {entry["url"] for entry in servers}:
        try:
            servers.append({"name": "", "url": stored["url"],
                            "models": refine_remote.list_models(
                                timeout=refine_remote.PROBE_TIMEOUT)})
        except refine.RefineError:
            pass
    return {
        "refiner": {
            "local": {"models": local, "model": _local_refiner(local)},
            "servers": servers,
            "stored": stored,
        },
        "families": chat.setup_report(manifest.catalog(), available,
                                      settings.load().get("weights") or {}),
    }


@PromptServer.instance.routes.get("/continuity/chat/setup")
async def chat_setup(request):
    """What the room's first run has to go on. See `_setup`.

    Off the event loop: it walks the model directories and knocks on two ports.
    """
    loop = asyncio.get_running_loop()
    try:
        return web.json_response(await loop.run_in_executor(None, _setup))
    except Exception as problem:  # noqa: BLE001
        return web.json_response({"error": f"{type(problem).__name__}: {problem}"},
                                 status=500)
