"""Throwaway socket.io smoke test for the Chainlit UI (not part of the test suite).

Connects to a running `chainlit run app.py` server, performs the handshake,
and exercises: on_chat_start messages/elements/settings, the
no-documents-selected guard, and the /documents command.
"""

import asyncio
import json
import uuid

import socketio

received = []  # (event, data) in arrival order

sio = socketio.AsyncClient()


@sio.on("*")
def catch_all(event, data=None):
    received.append((event, data))


async def wait_for(pred, timeout=15):
    for _ in range(timeout * 10):
        if pred():
            return True
        await asyncio.sleep(0.1)
    return False


def messages():
    return [
        d.get("output", "")
        for e, d in received
        if e == "new_message" and isinstance(d, dict)
    ]


def step_dict(content, name="User"):
    return {
        "id": uuid.uuid4().hex,
        "name": name,
        "type": "user_message",
        "output": content,
        "createdAt": "2026-01-01T00:00:00Z",
    }


async def main():
    await sio.connect(
        "http://localhost:8765",
        transports=["websocket"],
        socketio_path="/ws/socket.io",
        auth={
            "sessionId": uuid.uuid4().hex,
            "clientType": "webapp",
            "threadId": None,
            "userEnv": "{}",
            "chatProfile": None,
        },
    )
    await sio.emit("connection_successful")

    ok = await wait_for(lambda: any(e == "chat_settings" for e, _ in received))
    print("chat_settings received:", ok)
    settings_events = [d for e, d in received if e == "chat_settings"]
    if settings_events:
        payload = settings_events[0]
        inputs = payload.get("inputs", []) if isinstance(payload, dict) else payload
        print("settings widget types:", [i.get("type") for i in inputs])
        if inputs:
            print("multiselect id:", inputs[0].get("id"),
                  "| items:", len(inputs[0].get("items", [])))

    elements = [d for e, d in received if e == "element"]
    print("elements received:", [(el.get("type"), el.get("name")) for el in elements])

    print("chat_start messages:")
    for m in messages():
        print("  -", m[:90].replace("\n", " "))

    # Guard: ask a question with nothing selected.
    received.clear()
    await sio.emit("client_message", {"message": step_dict("What is this about?"), "fileReferences": None})
    ok = await wait_for(lambda: any("select at least one document" in m for m in messages()))
    print("guard message shown:", ok)

    # /documents command.
    received.clear()
    await sio.emit("client_message", {"message": step_dict("/documents"), "fileReferences": None})
    ok = await wait_for(lambda: any(e == "element" for e, _ in received) and len(messages()) >= 1)
    print("/documents produced table + actions:", ok)
    for m in messages():
        print("  -", m[:90].replace("\n", " "))
    action_msgs = [d for e, d in received if e == "new_message" and d.get("actions")]
    if action_msgs:
        print("delete actions:", [a.get("label") for a in action_msgs[0]["actions"]])

    # Select the first available document via settings, then ask a question.
    doc_items = []
    if settings_events:
        payload = settings_events[0]
        inputs = payload.get("inputs", []) if isinstance(payload, dict) else payload
        doc_items = inputs[0].get("items", []) if inputs else []
    if doc_items:
        doc_id = doc_items[0]["value"]
        received.clear()
        await sio.emit("chat_settings_change", {"documents": [doc_id]})
        ok = await wait_for(lambda: any("selected for chat" in m for m in messages()))
        print("settings update ack:", ok)

        received.clear()
        await sio.emit(
            "client_message",
            {"message": step_dict("What is this document about? Answer in one sentence."), "fileReferences": None},
        )
        # Tokens stream via "stream_token"; the final message arrives as "new_message"/update.
        ok = await wait_for(
            lambda: any(e == "stream_token" for e, _ in received), timeout=120
        )
        print("answer streamed tokens:", ok)
        tokens = [d.get("token", "") for e, d in received if e == "stream_token" and isinstance(d, dict)]
        answer = "".join(tokens)
        print("streamed answer (first 200 chars):", answer[:200].replace("\n", " "))
        ok2 = await wait_for(lambda: "Sources" in answer or len(answer) > 20, timeout=30)
        print("non-trivial answer received:", ok2)
    else:
        print("no processed documents available; skipping question test")

    await sio.disconnect()


asyncio.run(main())
