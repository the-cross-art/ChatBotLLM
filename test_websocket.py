#!/usr/bin/env python3

import asyncio
import websockets
import json


async def test_websocket():
    uri = "ws://localhost:8000/ws"

    try:
        print("🔌 Connecting to WebSocket...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected!")

            # Send init message
            init_message = {
                "user_id": "test_user_123",
                "thread_id": "test_user_123_abc123def456",
                "init": True,
            }
            print(f"📤 Sending init: {init_message}")
            await websocket.send(json.dumps(init_message))

            # Wait for init response
            response = await websocket.recv()
            print(f"📨 Received: {response}")

            # Send test chat message
            if "ok" in response:
                chat_message = {"message": "Hello, this is a test!"}
                print(f"📤 Sending chat: {chat_message}")
                await websocket.send(json.dumps(chat_message))

                # Receive responses
                for i in range(10):  # Listen for up to 10 responses
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        print(f"📨 Response {i+1}: {response}")

                        # Stop if we get final message
                        if "assistant_message" in response:
                            break

                    except asyncio.TimeoutError:
                        print("⏰ No more messages received")
                        break

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_websocket())
