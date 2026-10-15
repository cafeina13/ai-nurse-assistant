import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app import Demo, create_app
from backend.models import Event


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = tmp_path_factory.mktemp("demo") / "demo.db"
    app = create_app(demo_factory=lambda: Demo(db_path=db))
    with TestClient(app) as c:
        c.demo = app  # keep a handle for event checks
        yield c


def first_patient(client):
    return client.get("/api/state").json()["patients"][0]["id"]


def nurse(client):
    return client.get("/api/state").json()["nurses"][0]["id"]


def test_pages_are_served(client):
    assert "Hemşire Panosu" in client.get("/board").text
    assert "Hemşire Çağrı" in client.get("/patient").text


def test_critical_request_tops_the_queue_and_needs_confirmation(client):
    pid = first_patient(client)
    client.post("/api/requests", json={"patient_id": pid, "text": "Su alabilir miyim"})
    res = client.post("/api/requests", json={"patient_id": pid, "text": "Nefes alamıyorum"})
    # The patient hears the same calm message whatever the urgency.
    assert res.json()["message"] == "İsteğiniz hemşireye iletildi."

    queue = client.get("/api/state").json()["cards"]
    assert queue[0]["urgency"] == "critical" and queue[0]["text"] == "Nefes alamıyorum"
    assert any(r.startswith("rule:") for r in queue[0]["reasons"])

    card_id = queue[0]["id"]
    assert client.post(f"/api/requests/{card_id}/claim", json={"nurse_id": nurse(client)}).json()["status"] == "claimed"
    # Closing a critical asks whether it really was critical.
    assert client.post(f"/api/requests/{card_id}/complete", json={"nurse_id": nurse(client)}).status_code == 422
    done = client.post(f"/api/requests/{card_id}/complete",
                       json={"nurse_id": nurse(client), "really_critical": True})
    assert done.json()["status"] == "done"
    assert card_id not in [c["id"] for c in client.get("/api/state").json()["cards"]]


def test_buttons_skip_the_model(client):
    pid = first_patient(client)
    client.post("/api/requests", json={"patient_id": pid, "text": "[Su]", "channel": "button"})
    card = next(c for c in client.get("/api/state").json()["cards"] if c["text"] == "[Su]")
    assert card["cell"] == "comfort_supplies/routine" and card["confidence"] is None


def test_every_step_is_in_the_event_log(client):
    pid = first_patient(client)
    card_id = client.post("/api/requests", json={"patient_id": pid, "text": "Battaniye lazım"}).json()["id"]
    client.post(f"/api/requests/{card_id}/claim", json={"nurse_id": nurse(client)})
    client.post(f"/api/requests/{card_id}/complete", json={"nurse_id": nurse(client)})
    with client.app.state.demo.sessions() as session:
        types = [e.type for e in session.scalars(select(Event).order_by(Event.id))
                 if e.details.get("request_id") == card_id]
    assert types == ["request.created", "request.triaged", "request.claimed", "request.completed"]


def test_replay_fills_the_board_through_the_same_pipeline(client):
    assert client.post("/api/replay", json={"speed": 0}).json()["started"]
    task = client.app.state.demo.replay_task
    for _ in range(100):  # the replay runs in the background
        if task.done():
            break
        time.sleep(0.1)
    # Earlier tests typed requests first; that must not stop the replay.
    assert task.done() and task.exception() is None
    cards = client.get("/api/state").json()["cards"]
    assert sum(c["text"] != "[Su]" for c in cards) > 50
    assert all(c["reasons"] for c in cards)
    # The shift is over: the clock stops, so waiting times stop growing.
    for _ in range(50):
        if client.get("/api/state").json()["replay"] == "finished":
            break
        time.sleep(0.1)
    state = client.get("/api/state").json()
    assert state["replay"] == "finished"
    time.sleep(0.3)
    assert client.get("/api/state").json()["now"] == state["now"]
    # Requests typed before the replay share the ward's timeline, not the real date.
    assert all(c["created_at"][:4] == "2026" and c["created_at"][5:7] == "01" for c in state["cards"])
    # A ward can only be replayed once (the event log is append-only).
    assert client.post("/api/replay", json={"speed": 0}).status_code == 409
    assert client.post("/api/replay/control", json={"action": "pause"}).status_code == 409


def test_websocket_pushes_new_cards(client):
    pid = first_patient(client)
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "snapshot"
        client.post("/api/requests", json={"patient_id": pid, "text": "Ağrı kesici alabilir miyim"})
        msg = ws.receive_json()
        while msg["type"] != "card":
            msg = ws.receive_json()
        assert msg["card"]["text"] == "Ağrı kesici alabilir miyim"


def test_bad_requests_are_refused(client):
    assert client.post("/api/requests", json={"patient_id": 9999, "text": "su"}).status_code == 404
    assert client.post("/api/requests", json={"patient_id": first_patient(client), "text": "  "}).status_code == 422
    assert client.post("/api/requests/9999/claim", json={"nurse_id": 1}).status_code == 404
