"""The demo server: patient page -> triage -> live nurse board.

    uv run uvicorn backend.app:app --port 8000
    then open http://localhost:8000/board and http://localhost:8000/patient

Each start builds a fresh synthetic ward (the event log is append-only, so a replay cannot be
undone). Replayed calls use the Claude-written TEST texts, which the model never trained on,
so what the board shows is an honest picture of triage, not a rehearsal of its training data.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.board import Board
from backend.db import init_db, make_engine, session_factory
from backend.replay import Replay, SimClock
from triage.evaluate import load_examples
from triage.models import make_model
from triage.pipeline import Triage
from wardgen import DEFAULT_START, generate
from wardgen.textbank import TEXTBANK_DIR

REPO = Path(__file__).resolve().parent.parent
FRONTEND = REPO / "frontend"
DEMO_DB = REPO / "data" / "generated" / "demo.db"


class Clients:
    """Open board pages; every card change is pushed to all of them."""

    def __init__(self):
        self.sockets: set[WebSocket] = set()

    async def send(self, message: dict) -> None:
        for ws in list(self.sockets):
            try:
                await ws.send_json(message)
            except Exception:
                self.sockets.discard(ws)


class Demo:
    def __init__(self, db_path: Path = DEMO_DB, seed: int = 42):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.unlink(missing_ok=True)
        generate(db_path, seed=seed, days=1, textbank=TEXTBANK_DIR / "test" / "claude")
        engine = make_engine(db_path)
        init_db(engine)
        self.sessions = session_factory(engine)
        # The engine that won the head-to-head: TF-IDF + protocol (critical override) + safety
        # rules, trained on the training bank plus the tone bank.
        train = load_examples("train/gemini") + load_examples("train/gemini_tone")
        model = make_model("hybrid-critical:tfidf").fit([e.text for e in train], [e.cell for e in train])
        triage = Triage(model)
        # One simulated clock for everything: it runs in real time from the ward's start until a
        # replay speeds it up, and stops when the shift is over, so typed and replayed requests
        # always share one timeline.
        self.clock = SimClock(DEFAULT_START, speed=1.0)
        self.replay: Replay | None = None
        self.replay_task: asyncio.Task | None = None
        self.replay_state = "idle"  # idle | running | paused | finished
        self.board = Board(self.sessions, triage, self.now)
        self.clients = Clients()

    def now(self) -> datetime:
        return self.clock.now().replace(microsecond=0)

    async def push_status(self) -> None:
        await self.clients.send({"type": "status", "replay": self.replay_state,
                                 "speed": self.clock.speed, "now": self.now().isoformat()})

    async def push(self, card: dict) -> None:
        await self.clients.send({"type": "card", "card": card, "now": self.now().isoformat()})

    async def start_replay(self, speed: float) -> None:
        if self.replay is not None:
            raise HTTPException(409, "this ward has already been replayed; restart the server for a new one")
        self.clock.set_speed(speed or None)  # 0 = as fast as possible
        self.replay = Replay(self.sessions, clock=self.clock)

        async def on_event(event):
            if event.type == "request.created":
                await self.push(self.board.on_created(event))

        async def run():
            await self.replay.run()
            self.clock.pause()  # the shift is over: time stops, waiting times stop growing
            self.replay_state = "finished"
            await self.push_status()

        self.replay.subscribe(on_event)
        self.replay_state = "running"
        self.replay_task = asyncio.create_task(run())
        await self.push_status()

    async def control(self, action: str, speed: float | None = None) -> None:
        if self.replay_state in ("idle", "finished"):
            raise HTTPException(409, f"no replay running ({self.replay_state})")
        if action == "pause":
            self.clock.pause()
            self.replay_state = "paused"
        elif action == "resume":
            self.clock.resume()
            self.replay_state = "running"
        elif action == "speed":
            if not speed or speed <= 0:
                raise HTTPException(422, "speed must be positive")
            self.clock.set_speed(speed)
        await self.push_status()


class NewRequest(BaseModel):
    patient_id: int
    text: str
    channel: str = "text"  # text | button
    speaker: str = "patient"


class Action(BaseModel):
    nurse_id: int
    really_critical: bool | None = None


class ReplayStart(BaseModel):
    speed: float = 600.0  # simulated seconds per real second: 10 sim minutes per second


class ReplayControl(BaseModel):
    action: str  # pause | resume | speed
    speed: float | None = None


def create_app(demo_factory=Demo) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["demo"] = app.state.demo = demo_factory()
        yield
        task = state["demo"].replay_task
        if task:
            task.cancel()

    app = FastAPI(title="AI Nurse Assistant demo", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    def demo() -> Demo:
        return state["demo"]

    @app.get("/")
    def index():
        return RedirectResponse("/board")

    @app.get("/board")
    def board_page():
        return FileResponse(FRONTEND / "board.html")

    @app.get("/patient")
    def patient_page():
        return FileResponse(FRONTEND / "patient.html")

    @app.get("/api/state")
    def get_state():
        d = demo()
        return {"cards": d.board.queue(), "nurses": d.board.nurses(), "patients": d.board.patients(),
                "now": d.now().isoformat(), "replay": d.replay_state, "speed": d.clock.speed}

    @app.post("/api/requests")
    async def new_request(body: NewRequest):
        if not body.text.strip():
            raise HTTPException(422, "empty request")
        try:
            card = demo().board.create_request(body.patient_id, body.text.strip(), body.channel, body.speaker)
        except KeyError as e:
            raise HTTPException(404, str(e))
        await demo().push(card)
        # Never tell a patient their request is not urgent: the answer is the same for all.
        return {"id": card["id"], "message": "İsteğiniz hemşireye iletildi."}

    @app.post("/api/requests/{request_id}/claim")
    async def claim(request_id: int, body: Action):
        try:
            card = demo().board.claim(request_id, body.nurse_id)
        except KeyError as e:
            raise HTTPException(404, str(e))
        await demo().push(card)
        return card

    @app.post("/api/requests/{request_id}/complete")
    async def complete(request_id: int, body: Action):
        try:
            card = demo().board.complete(request_id, body.nurse_id, body.really_critical)
        except KeyError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(422, str(e))
        await demo().push(card)
        return card

    @app.post("/api/replay")
    async def replay(body: ReplayStart):
        await demo().start_replay(body.speed)
        return {"started": True, "speed": body.speed}

    @app.post("/api/replay/control")
    async def replay_control(body: ReplayControl):
        await demo().control(body.action, body.speed)
        return {"replay": demo().replay_state, "speed": demo().clock.speed}

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        d = demo()
        d.clients.sockets.add(socket)
        await socket.send_json({"type": "snapshot", "cards": d.board.queue(), "now": d.now().isoformat(),
                                "replay": d.replay_state, "speed": d.clock.speed})
        try:
            while True:  # the board only listens; a tick keeps waiting times moving
                await asyncio.sleep(1)
                await socket.send_json({"type": "clock", "now": d.now().isoformat()})
        except (WebSocketDisconnect, RuntimeError):
            d.clients.sockets.discard(socket)

    return app


app = create_app()
